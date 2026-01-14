"""Azure OpenAI Realtime Voice API client via WebSocket."""

from __future__ import annotations

import aiohttp
import base64
import json
import ssl
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Callable
from urllib.parse import urlencode

from LingoGPTConnector.config import Config

logger = logging.getLogger(__name__)


@dataclass
class RealtimeSessionParams:
    """Parameters for configuring the Realtime session."""
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    turn_detection: Optional[Dict[str, Any]] = field(default_factory=lambda: {"type": "server_vad"})
    input_audio_transcription: Optional[Dict[str, Any]] = None
    extra_session: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealtimeResponseParams:
    """Parameters for response.create events."""
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    instructions: Optional[str] = None
    extra_response: Dict[str, Any] = field(default_factory=dict)


class RealtimeVoiceClient:
    """Client for connecting to Azure OpenAI Realtime API via WebSocket."""

    def __init__(self, config: Config, ssl_verify: bool = True):
        self.config = config
        self.ssl_verify = ssl_verify

        if not ssl_verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self._ssl_context = ctx
        else:
            self._ssl_context = None

    def _build_ws_url(self, cfg: dict) -> str:
        """
        Build the complete WebSocket URL for Azure OpenAI Realtime API.

        Format: wss://{resource}.openai.azure.com/openai/realtime?api-version={version}&deployment={deployment}
        """
        api_url = cfg.get("api_url", "")

        # Convert https:// to wss://
        if api_url.startswith("https://"):
            ws_base = "wss://" + api_url[len("https://"):]
        elif api_url.startswith("http://"):
            ws_base = "ws://" + api_url[len("http://"):]
        else:
            ws_base = api_url

        # Ensure no trailing slash
        ws_base = ws_base.rstrip("/")

        # Build query parameters
        params = {
            "api-version": cfg.get("api_version"),
            "deployment": cfg.get("deployment"),
        }
        query_string = urlencode(params)

        # Construct full URL
        ws_url = f"{ws_base}/openai/realtime?{query_string}"
        logger.debug(f"Built WebSocket URL: {ws_url}")
        return ws_url

    def _build_headers(self, cfg: dict) -> Dict[str, str]:
        """Build authentication headers for WebSocket connection."""
        headers: Dict[str, str] = {}

        if cfg["resource_flag"] == "KEY":
            api_key = cfg.get("api_key")
            if not api_key:
                raise ValueError("Realtime: api_key missing for resource_flag=KEY")
            headers["api-key"] = api_key
        else:
            # TOKEN mode - use OAuth bearer token
            token = self.config.get_openai_token()
            headers["Authorization"] = f"Bearer {token}"

            project_id = cfg.get("project_id")
            if project_id:
                headers["projectId"] = str(project_id)

        return headers

    async def connect(
        self,
        *,
        session_params: Optional[RealtimeSessionParams] = None,
        response_params: Optional[RealtimeResponseParams] = None,
    ) -> "RealtimeConnection":
        """
        Establish a WebSocket connection to the Realtime API.

        Returns a RealtimeConnection that can be used to send/receive events.
        """
        cfg = self.config.get_realtime_config()

        # Validate required config
        if not cfg.get("api_url"):
            raise ValueError("Realtime: api_url missing in config.")
        if not cfg.get("deployment"):
            raise ValueError("Realtime: deployment missing in config. Set LINGO_REALTIME_DEPLOYMENT.")
        if not cfg.get("api_version"):
            raise ValueError("Realtime: api_version missing in config.")

        # Build the full WebSocket URL
        ws_url = self._build_ws_url(cfg)
        headers = self._build_headers(cfg)

        logger.info(f"Connecting to Realtime API at: {ws_url}")

        # Create HTTP session and connect
        http_session = aiohttp.ClientSession()
        try:
            ws = await http_session.ws_connect(
                ws_url,
                headers=headers,
                ssl=self._ssl_context,
                heartbeat=30,
            )
        except Exception as e:
            await http_session.close()
            raise RuntimeError(f"Failed to connect to Realtime API: {e}") from e

        conn = RealtimeConnection(
            http_session=http_session,
            ws=ws,
            session_params=session_params or RealtimeSessionParams(),
            response_params=response_params or RealtimeResponseParams(),
        )

        # Wait for session.created event
        logger.info("Waiting for session.created event...")
        async for event in conn.iter_events():
            event_type = event.get("type")
            logger.debug(f"Received event: {event_type}")
            if event_type == "session.created":
                logger.info("Session created successfully")
                break
            elif event_type == "error":
                raise RuntimeError(f"Error during session creation: {event}")

        # Send session.update to configure the session
        await conn.session_update()
        return conn


class RealtimeConnection:
    """Active connection to the Realtime API."""

    def __init__(
        self,
        *,
        http_session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        session_params: RealtimeSessionParams,
        response_params: RealtimeResponseParams,
    ):
        self.http_session = http_session
        self.ws = ws
        self.session_params = session_params
        self.response_params = response_params

    async def close(self) -> None:
        """Close the WebSocket connection and HTTP session."""
        try:
            await self.ws.close()
        finally:
            await self.http_session.close()

    async def send_event(self, event: Dict[str, Any]) -> None:
        """Send a JSON event to the server."""
        event_json = json.dumps(event)
        logger.debug(f"Sending event: {event.get('type')}")
        await self.ws.send_str(event_json)

    async def session_update(self) -> None:
        """
        Send session.update to configure the session.

        IMPORTANT: Must include 'modalities' for audio output to work!
        """
        session_obj: Dict[str, Any] = {
            # CRITICAL: modalities must be set for audio output
            "modalities": ["text", "audio"],
            "turn_detection": self.session_params.turn_detection,
            "voice": self.session_params.voice,
            "input_audio_format": self.session_params.input_audio_format,
            "output_audio_format": self.session_params.output_audio_format,
        }

        if self.session_params.input_audio_transcription:
            session_obj["input_audio_transcription"] = self.session_params.input_audio_transcription

        if self.session_params.extra_session:
            session_obj.update(self.session_params.extra_session)

        logger.info(f"Sending session.update with modalities: {session_obj['modalities']}")
        await self.send_event({"type": "session.update", "session": session_obj})

    async def input_audio_append(self, pcm_bytes: bytes) -> None:
        """Append PCM16 audio data to the input buffer."""
        await self.send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_bytes).decode("utf-8"),
            }
        )

    async def input_audio_commit(self) -> None:
        """Commit the audio buffer, signaling end of user speech (manual mode)."""
        await self.send_event({"type": "input_audio_buffer.commit"})

    async def input_audio_clear(self) -> None:
        """Clear the input audio buffer."""
        await self.send_event({"type": "input_audio_buffer.clear"})

    async def response_create(self, *, override: Optional[RealtimeResponseParams] = None) -> None:
        """
        Trigger a response from the model.

        In server_vad mode, this is called automatically when speech ends.
        In manual mode (turn_detection=None), call this after input_audio_commit().
        """
        rp = override or self.response_params

        resp_obj: Dict[str, Any] = {
            "modalities": rp.modalities,
            # Include voice and output format for audio responses
            "voice": self.session_params.voice,
            "output_audio_format": self.session_params.output_audio_format,
        }

        if rp.instructions:
            resp_obj["instructions"] = rp.instructions

        if rp.extra_response:
            resp_obj.update(rp.extra_response)

        logger.info("Sending response.create")
        await self.send_event({"type": "response.create", "response": resp_obj})

    async def response_cancel(self) -> None:
        """Cancel the current response."""
        await self.send_event({"type": "response.cancel"})

    async def iter_events(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Iterate over all events from the server.

        Yields each event as a dictionary. Stops when connection closes.
        """
        while True:
            try:
                msg = await self.ws.receive()
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break

            if msg.type == aiohttp.WSMsgType.CLOSED:
                logger.info("WebSocket closed")
                break

            if msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {msg}")
                raise RuntimeError(f"WebSocket error: {msg}")

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                event = json.loads(msg.data)
                yield event
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse event JSON: {e}")
                continue

    async def iter_audio_deltas(self) -> AsyncIterator[bytes]:
        """
        Iterate over audio delta events, yielding decoded PCM bytes.

        Handles both Azure naming conventions:
        - response.audio.delta / response.audio.done
        - response.output_audio.delta / response.output_audio.done
        """
        async for event in self.iter_events():
            event_type = event.get("type")

            if event_type in ("response.audio.delta", "response.output_audio.delta"):
                delta = event.get("delta")
                if delta:
                    yield base64.b64decode(delta)

            elif event_type in ("response.audio.done", "response.output_audio.done"):
                logger.info("Audio response complete")
                return

            elif event_type == "response.done":
                logger.info("Response complete")
                return

            elif event_type == "error":
                logger.error(f"Server error: {event}")
                raise RuntimeError(event)

    async def run_conversation(
        self,
        *,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_transcript: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Main event loop for processing all events.

        Args:
            on_event: Called for every event (for debugging)
            on_audio: Called with decoded audio bytes from response.audio.delta
            on_transcript: Called with (role, text) for transcript events
        """
        async for event in self.iter_events():
            event_type = event.get("type", "")

            # Call the generic event handler
            if on_event:
                on_event(event)

            # Handle audio deltas
            if event_type in ("response.audio.delta", "response.output_audio.delta"):
                if on_audio:
                    delta = event.get("delta")
                    if delta:
                        on_audio(base64.b64decode(delta))

            # Handle transcripts
            elif event_type == "response.audio_transcript.delta":
                if on_transcript:
                    delta = event.get("delta", "")
                    on_transcript("assistant", delta)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                if on_transcript:
                    transcript = event.get("transcript", "")
                    on_transcript("user", transcript)

            # Handle errors
            elif event_type == "error":
                logger.error(f"Server error: {event}")
                raise RuntimeError(event)

    async def debug_print_events(self, max_events: int = 200) -> None:
        """
        Debug utility: print all events from the server.

        Useful for diagnosing issues with the connection.
        """
        i = 0
        async for event in self.iter_events():
            i += 1
            event_type = event.get("type")
            print(f"[{i}] EVENT: {event_type}")

            if event_type in ("response.audio.delta", "response.output_audio.delta"):
                print(f"     audio_b64_len: {len(event.get('delta', ''))}")

            if event_type in ("response.audio.done", "response.output_audio.done", "response.done"):
                print("     [response complete]")

            if event_type == "error":
                print(f"     ERROR: {event}")
                break

            if event_type in ("input_audio_buffer.speech_started",):
                print("     [VAD detected speech start]")

            if event_type in ("input_audio_buffer.speech_stopped",):
                print("     [VAD detected speech end]")

            if i >= max_events:
                print(f"Reached max_events ({max_events})")
                break
