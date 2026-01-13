import asyncio
import base64
import io
import wave
from datetime import datetime

import streamlit as st

from LingoGPTConnector.config import Config
from LingoGPTConnector.client import GPTClient  # your existing GPTClient


# ---------------------------
# Helpers
# ---------------------------

def wav_bytes_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """
    Returns (pcm16_bytes, sample_rate, channels).
    Requires WAV with 16-bit PCM. If your st.audio_input uses a different format,
    you may need a decode step (but Streamlit's audio_input returns WAV bytes).
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV (sampwidth=2). Got sampwidth={sampwidth}.")
    if channels not in (1, 2):
        raise ValueError(f"Expected 1 or 2 channels. Got {channels}.")

    # If stereo, keep as-is; Azure realtime generally expects mono pcm16.
    # Best practice: enforce mono in st.audio_input by recording mono when possible,
    # or downmix here if needed.
    return frames, sample_rate, channels


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return bio.getvalue()


def run_async(coro):
    """
    Run an async coroutine from Streamlit (which is sync).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If there's already a running loop (rare in Streamlit), make a new one.
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    else:
        return asyncio.run(coro)


async def realtime_roundtrip_audio(
    gpt: GPTClient,
    pcm_in: bytes,
    *,
    input_chunk_bytes: int = 3200,   # ~100ms at 16kHz mono PCM16
    instructions: str | None = None,
) -> tuple[bytes, str | None]:
    """
    Sends PCM16 audio to realtime model and returns (assistant_pcm16_bytes, transcript_if_any).

    Assumes your GPTClient has:
        conn = await gpt.realtime_voice(...)
    and conn has:
        input_audio_append, input_audio_commit, response_create, iter_events/iter_audio_deltas, close
    """
    # Push-to-talk mode: server_vad=False so we explicitly commit
    conn = await gpt.realtime_voice(server_vad=False, instructions=instructions)

    assistant_audio = bytearray()
    transcript = None

    try:
        # 1) Append mic audio in chunks
        for i in range(0, len(pcm_in), input_chunk_bytes):
            chunk = pcm_in[i : i + input_chunk_bytes]
            await conn.input_audio_append(chunk)

        # 2) Commit (turn audio buffer into a conversation item)
        await conn.input_audio_commit()

        # 3) Ask for response (text+audio)
        await conn.response_create()

        # 4) Read events until response.done; collect response.audio.delta
        async for event in conn.iter_events():
            t = event.get("type")

            if t == "response.audio.delta":
                assistant_audio.extend(base64.b64decode(event["delta"]))

            elif t == "response.audio.done":
                # audio finished for this response; keep waiting for response.done
                pass

            elif t == "response.done":
                # Try to pull transcript if present
                try:
                    transcript = event["response"]["output"][0]["content"][0].get("transcript")
                except Exception:
                    transcript = None
                break

            elif t == "error":
                raise RuntimeError(event)

    finally:
        await conn.close()

    return bytes(assistant_audio), transcript


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Lingo Realtime Voice Demo (WebSocket)")

st.caption(
    "Records audio in the browser, streams it to Azure Realtime over websockets, "
    "and plays the assistant audio reply."
)

# Streamlit built-in mic recorder :contentReference[oaicite:2]{index=2}
audio_file = st.audio_input("Record a question", sample_rate=16000)

instructions = st.text_area(
    "Optional instructions (sent to the model)",
    value="Answer clearly and briefly.",
    height=80,
)

if audio_file is not None:
    wav_bytes = audio_file.read()
    st.audio(wav_bytes, format="audio/wav")  # playback user audio :contentReference[oaicite:3]{index=3}

    if st.button("Send to realtime model"):
        with st.spinner("Streaming to model..."):
            try:
                # Convert WAV -> PCM16 bytes
                pcm_in, sr, ch = wav_bytes_to_pcm16(wav_bytes)

                # (Optional) warn if not 16k mono
                if sr != 16000:
                    st.warning(f"Your mic recording sample rate is {sr}Hz (expected 16000Hz).")
                if ch != 1:
                    st.warning(f"Your mic recording has {ch} channels (expected mono).")

                config = Config()
                gpt = GPTClient(config)

                assistant_pcm, transcript = run_async(
                    realtime_roundtrip_audio(
                        gpt,
                        pcm_in,
                        instructions=instructions.strip() or None,
                    )
                )

                # Most realtime outputs are pcm16; commonly 24kHz for output.
                # Use your realtime session config/output format rate if different.
                assistant_wav = pcm16_to_wav_bytes(assistant_pcm, sample_rate=24000, channels=1)

                st.success("Got response!")
                if transcript:
                    st.markdown("### Transcript")
                    st.write(transcript)

                st.markdown("### Assistant audio")
                st.audio(assistant_wav, format="audio/wav")  # :contentReference[oaicite:4]{index=4}

                # Let user download
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "Download assistant reply WAV",
                    data=assistant_wav,
                    file_name=f"assistant_reply_{ts}.wav",
                    mime="audio/wav",
                )

            except Exception as e:
                st.error(f"Error: {e}")
