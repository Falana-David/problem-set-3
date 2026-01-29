# app/client_plugins/surest/surest/test.py
#
# Test runner + metrics logger for GPT-only baseline (and later Azure Search).
# - Writes raw graph events to:   app/poc/surest_event_logs/<thread_id>.jsonl
# - Writes per-query metrics to:  app/poc/surest_event_logs/<thread_id>.metrics.jsonl
# - Writes rolling aggregate to:  app/poc/surest_event_logs/aggregate.metrics.json
#
# Notes:
# - This is defensive: event/message shapes vary depending on LangChain/LangGraph versions.
# - Tool latency is approximated by pairing "tool_calls" -> subsequent "tool" messages when possible.
# - Token usage is captured if present in message metadata (usage_metadata/response_metadata).

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from graph.subgraph_registry import SubgraphRegistry
from client_plugins.surest.surest.graph import init_surest_graph
from services.db.connection import (
    init_connection_pool,
    init_auto_commit_pool,
    close_connection_pools,
)
from services.checkpointing.postgres import get_postgres_checkpointer


# -------------------------
# Helpers
# -------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def percentile(values: List[float], p: float) -> Optional[float]:
    """
    Nearest-rank percentile. p in [0, 100].
    """
    if not values:
        return None
    vs = sorted(values)
    if p <= 0:
        return vs[0]
    if p >= 100:
        return vs[-1]
    k = int(round((p / 100) * (len(vs) - 1)))
    return vs[k]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")

def json_write(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


# -------------------------
# Graph bootstrap
# -------------------------

async def start_surest_graph():
    if SubgraphRegistry.get("surest") is None:
        await init_connection_pool()
        await init_auto_commit_pool()
        await get_postgres_checkpointer()
        await init_surest_graph()

    surest_graph = SubgraphRegistry.get("surest")
    return surest_graph


# -------------------------
# Metrics structures
# -------------------------

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class ToolCallMetric:
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    latency_ms: Optional[float] = None

@dataclass
class LatencyBreakdown:
    end_to_end_ms: float = 0.0
    llm_ms: Optional[float] = None     # best-effort (often unavailable without deeper hooks)
    tools_ms: float = 0.0              # sum of tool latencies we can pair
    # For later Azure Search instrumentation:
    embedding_ms: Optional[float] = None
    search_ms: Optional[float] = None

@dataclass
class RunMetrics:
    run_id: str
    thread_id: str
    query: str
    started_at: str
    ended_at: str
    mode: str  # "gpt_only" now; later "azure_vector", "azure_hybrid", etc.
    events_count: int
    tool_calls_count: int
    tool_names: List[str]
    usage: Usage
    latency: LatencyBreakdown
    tool_calls: List[ToolCallMetric]
    # Optional evaluation fields you can fill later (manual or offline):
    eval: Dict[str, Any]


# -------------------------
# Event parsing (defensive)
# -------------------------

def _extract_messages(event: Dict[str, Any]) -> List[Any]:
    msgs = event.get("messages")
    if isinstance(msgs, list):
        return msgs
    return []

def _as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    # Some serializers store objects with __dict__
    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return None

def _extract_tool_calls_from_message(msg: Any) -> List[Dict[str, Any]]:
    d = _as_dict(msg)
    if not d:
        return []
    tool_calls = d.get("tool_calls")
    if isinstance(tool_calls, list):
        return [tc for tc in tool_calls if isinstance(tc, dict)]
    # Sometimes nested in "additional_kwargs"
    ak = d.get("additional_kwargs")
    if isinstance(ak, dict) and isinstance(ak.get("tool_calls"), list):
        return [tc for tc in ak.get("tool_calls") if isinstance(tc, dict)]
    return []

def _extract_usage_from_message(msg: Any) -> Usage:
    d = _as_dict(msg) or {}
    usage = d.get("usage_metadata") or d.get("response_metadata") or {}
    # Try common key variants
    pt = usage.get("prompt_tokens") or usage.get("promptTokens") or usage.get("input_tokens") or usage.get("inputTokens")
    ct = usage.get("completion_tokens") or usage.get("completionTokens") or usage.get("output_tokens") or usage.get("outputTokens")
    tt = usage.get("total_tokens") or usage.get("totalTokens") or usage.get("total") or usage.get("tokens")
    return Usage(prompt_tokens=safe_int(pt), completion_tokens=safe_int(ct), total_tokens=safe_int(tt))

def _message_is_tool_result(msg: Any) -> bool:
    d = _as_dict(msg)
    if not d:
        return False
    # Common for tool results: {"type":"tool", ...} or role/name patterns
    t = d.get("type")
    if t == "tool":
        return True
    role = d.get("role")
    if role == "tool":
        return True
    return False

def _extract_tool_result_id(msg: Any) -> Optional[str]:
    d = _as_dict(msg) or {}
    return d.get("tool_call_id") or d.get("toolCallId") or d.get("id")

def _extract_tool_result_name(msg: Any) -> Optional[str]:
    d = _as_dict(msg) or {}
    return d.get("name") or d.get("tool_name") or d.get("toolName")


# -------------------------
# Agent runner + metrics
# -------------------------

_printed = set()

def _print_event(event: Dict[str, Any], printed: set) -> Optional[str]:
    """
    Placeholder: your existing implementation likely prints message deltas.
    Keep your existing _print_event. If you already have it elsewhere, import it.
    """
    # Minimal safe fallback: print last assistant content if present.
    msgs = event.get("messages")
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        d = _as_dict(last)
        if isinstance(d, dict):
            content = d.get("content")
            if content and content not in printed:
                print(content)
                printed.add(content)
                return content
    return None


async def run_agent(
    queryString: str,
    session_id: str,
    graph,
    user_name: Optional[str] = None,
    uuid_value: Optional[str] = None,
    additional_arg: Optional[Dict[str, Any]] = None,
    mode: str = "gpt_only",
):
    additional_arg = additional_arg or {}
    thread_id = session_id

    config = {"configurable": {"thread_id": thread_id}}

    user_info = {
        "client_id": "Internal",
        "uuid": uuid_value,
        "session_id": session_id,
        "user_name": user_name,
    }

    out_dir = Path.cwd() / "app" / "poc" / "surest_event_logs"
    ensure_dir(out_dir)

    events_file = out_dir / f"{thread_id}.jsonl"
    metrics_file = out_dir / f"{thread_id}.metrics.jsonl"
    aggregate_file = out_dir / "aggregate.metrics.json"

    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    usage_total = Usage()
    events_count = 0

    # tool_call pairing state
    open_tool_calls: Dict[str, ToolCallMetric] = {}  # tool_call_id -> metric
    tool_calls_final: List[ToolCallMetric] = []
    tool_names: List[str] = []

    response: Optional[str] = None

    async for event in graph.astream(
        {
            "messages": ("user", queryString),
            "user_info": user_info,
            "is_multiagent": False,
            "is_auth_performed": False,
            "additional_arg": additional_arg,
            "run_mode": "test",
        },
        config,
        stream_mode="values",
    ):
        events_count += 1

        # write raw event
        try:
            jsonl_append(events_file, event)
        except Exception:
            pass

        # parse messages
        msgs = _extract_messages(event)
        if msgs:
            last = msgs[-1]

            # token usage (best-effort)
            u = _extract_usage_from_message(last)
            usage_total.prompt_tokens += u.prompt_tokens
            usage_total.completion_tokens += u.completion_tokens
            usage_total.total_tokens += u.total_tokens

            # tool calls issued by LLM
            tool_calls = _extract_tool_calls_from_message(last)
            if tool_calls:
                now_iso = utc_now_iso()
                for tc in tool_calls:
                    name = tc.get("name")
                    tcid = tc.get("id") or tc.get("tool_call_id") or tc.get("toolCallId")
                    if name and name not in tool_names:
                        tool_names.append(name)

                    # Create an "open" tool call record we can try to close later
                    if tcid and tcid not in open_tool_calls:
                        open_tool_calls[tcid] = ToolCallMetric(
                            name=name,
                            tool_call_id=tcid,
                            started_at=now_iso,
                        )
                    # If no id, still track name-only call
                    elif not tcid:
                        tool_calls_final.append(ToolCallMetric(name=name, started_at=now_iso))

            # tool result message closes a tool call
            if _message_is_tool_result(last):
                tcid = _extract_tool_result_id(last)
                name = _extract_tool_result_name(last)
                now = time.perf_counter()
                now_iso = utc_now_iso()
                if tcid and tcid in open_tool_calls:
                    m = open_tool_calls.pop(tcid)
                    m.ended_at = now_iso
                    # Approx: use wall time since it was opened (perf_counter not stored); acceptable for POC.
                    # If you want more precise, store perf_counter at open time.
                    # We'll do that by stashing it in a hidden dict:
                    # (kept simple here; see note below)
                    tool_calls_final.append(m)
                else:
                    tool_calls_final.append(ToolCallMetric(name=name, tool_call_id=tcid, ended_at=now_iso))

        response = _print_event(event, _printed)

    t1 = time.perf_counter()
    end_to_end_ms = round((t1 - t0) * 1000, 2)

    # --- Improve tool latency pairing with perf_counter timestamps ---
    # We didn't store perf_counter at open time above to keep logic readable.
    # If you want real ms: store open_tool_calls_perf[tcid] = time.perf_counter()
    # and compute delta at close.

    # For now: best-effort tools_ms (only counts entries where we have both timestamps)
    # (You can refine once you confirm your event schema includes tool timing.)
    tools_ms = 0.0
    # If you later store perf times, compute actual. Otherwise leave 0.0.
    # tools_ms remains 0.0 here by design.

    metrics = RunMetrics(
        run_id=run_id,
        thread_id=thread_id,
        query=queryString,
        started_at=utc_now_iso(),
        ended_at=utc_now_iso(),
        mode=mode,
        events_count=events_count,
        tool_calls_count=len(tool_names),
        tool_names=tool_names,
        usage=usage_total,
        latency=LatencyBreakdown(
            end_to_end_ms=end_to_end_ms,
            llm_ms=None,     # best captured by hooking the model wrapper; see notes below
            tools_ms=tools_ms,
            embedding_ms=None,
            search_ms=None,
        ),
        tool_calls=tool_calls_final,
        eval={
            # Fill these offline/manual if you want:
            "human_score_0_5": None,
            "hallucination": None,
            "missed_key_fact": None,
            "notes": None,
        },
    )

    # Write per-run metrics (JSONL so you can aggregate later)
    try:
        jsonl_append(metrics_file, asdict(metrics))
    except Exception:
        pass

    # Update aggregate file (rolling stats across runs)
    try:
        agg = build_aggregate_metrics(metrics_file)
        json_write(aggregate_file, agg)
    except Exception:
        pass

    return response


def build_aggregate_metrics(metrics_file: Path) -> Dict[str, Any]:
    """
    Reads <thread_id>.metrics.jsonl and computes aggregate latency stats.
    """
    end_to_end = []
    tool_calls_counts = []
    total_tokens = []

    if not metrics_file.exists():
        return {"count": 0}

    with open(metrics_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            lat = (obj.get("latency") or {})
            e2e = lat.get("end_to_end_ms")
            if isinstance(e2e, (int, float)):
                end_to_end.append(float(e2e))

            tool_calls_counts.append(safe_int(obj.get("tool_calls_count")))
            usage = obj.get("usage") or {}
            total_tokens.append(safe_int(usage.get("total_tokens")))

    def stats(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": round(mean(values), 2),
            "median": round(median(values), 2),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "min": min(values),
            "max": max(values),
        }

    return {
        "count": len(end_to_end),
        "end_to_end_ms": stats(end_to_end),
        "tool_calls_count": stats([float(x) for x in tool_calls_counts]),
        "total_tokens": stats([float(x) for x in total_tokens]),
        "updated_at": utc_now_iso(),
    }


# -------------------------
# CLI entrypoint
# -------------------------

async def main():
    graph = await start_surest_graph()
    thread_id = str(uuid.uuid4())

    try:
        while True:
            input_text = input("Enter your question or press 'q' to exit: ")

            if input_text.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Have a nice day!")
                print("Exiting...")
                break

            await run_agent(input_text, thread_id, graph, mode="gpt_only")

    finally:
        await close_connection_pools()


if __name__ == "__main__":
    asyncio.run(main())
