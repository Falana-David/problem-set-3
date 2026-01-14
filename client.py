"""GPT Client for Azure OpenAI services."""

import base64
import os
from io import BytesIO
from typing import Optional, Union, List

import pdfplumber
from pypdf import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from openai import AzureOpenAI

from LingoGPTConnector.config import Config
from LingoGPTConnector.realtime_voice import (
    RealtimeVoiceClient,
    RealtimeSessionParams,
    RealtimeResponseParams,
)


def _has_form_fields(pdf_bytes: bytes) -> bool:
    """Check if PDF has fillable form fields."""
    pdf = PdfReader(BytesIO(pdf_bytes))
    for page in pdf.pages:
        if "/Annots" in page:
            for annotation in page["/Annots"]:
                annot_object = pdf.get_object(annotation)
                if annot_object.get("/Subtype") == "/Widget":
                    return True
    return False


def _flatten_pdf_forms(pdf_file_like) -> bytes:
    """Flatten PDF form fields into static content."""
    input_pdf = PdfReader(pdf_file_like)
    output_pdf = PdfWriter()

    for page in input_pdf.pages:
        packet = BytesIO()
        c = canvas.Canvas(
            packet,
            pagesize=(page.mediabox.upper_right[0], page.mediabox.upper_right[1]),
        )

        if "/Annots" in page:
            for annotation in page["/Annots"]:
                annot_object = input_pdf.get_object(annotation)
                if "/T" in annot_object and "/V" in annot_object:
                    value = annot_object["/V"]
                    if value in ("/Off", None):
                        continue
                    if value == "/Yes":
                        value = "x"

                    rect = annot_object.get("/Rect")
                    if rect:
                        x, y = float(rect[0]), float(rect[1])
                        y += (float(rect[3]) - float(rect[1])) / 2
                        c.setFont("Helvetica", 12)
                        c.drawString(x, y, str(value))

        c.save()
        packet.seek(0)

        overlay_pdf = PdfReader(packet)
        if overlay_pdf.pages:
            page.merge_page(overlay_pdf.pages[0])

        output_pdf.add_page(page)

    output = BytesIO()
    output_pdf.write(output)
    output.seek(0)
    return output.getvalue()


def _convert_pdf_to_base64_images(pdf_bytes: bytes) -> List[str]:
    """Convert PDF pages to base64-encoded PNG images."""
    base64_images = []
    flattened = _flatten_pdf_forms(BytesIO(pdf_bytes)) if _has_form_fields(pdf_bytes) else pdf_bytes

    with pdfplumber.open(BytesIO(flattened)) as pdf:
        for page in pdf.pages:
            page_image = page.to_image(resolution=200)
            pil_image = page_image.original
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            base64_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return base64_images


class GPTClient:
    """Client for Azure OpenAI GPT services."""

    def __init__(self, config: Config):
        self.config = config

    def invoke_whisper(self, audio_file=None, audio_path: Optional[str] = None):
        """
        Transcribe audio using Azure OpenAI Whisper.

        Args:
            audio_file: File-like object containing audio data
            audio_path: Path to audio file on disk

        Returns:
            Transcription result from Whisper
        """
        cfg = self.config.get_whisper_config()

        if cfg["resource_flag"] != "KEY":
            raise ValueError("Whisper is only supported in KEY mode.")

        client = AzureOpenAI(
            azure_endpoint=cfg["api_url"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
        )

        if audio_file:
            return client.audio.transcriptions.create(file=audio_file, model=cfg["deployment"])
        if audio_path:
            with open(audio_path, "rb") as f:
                return client.audio.transcriptions.create(file=f, model=cfg["deployment"])

        raise ValueError("Must provide audio_file or audio_path.")

    def invoke_embedding(self, data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text using Azure OpenAI.

        Args:
            data: Single string or list of strings to embed

        Returns:
            Single embedding vector or list of embedding vectors
        """
        cfg = self.config.get_embedding_config()

        if cfg["resource_flag"] == "KEY":
            client = AzureOpenAI(
                azure_endpoint=cfg["api_url"],
                api_key=cfg["api_key"],
                api_version=cfg["api_version"],
            )
        else:
            client = AzureOpenAI(
                azure_endpoint=cfg["api_url"],
                azure_ad_token_provider=self.config.get_openai_token,
                api_version=cfg["api_version"],
                default_headers={"projectId": cfg["project_id"]},
            )

        result = client.embeddings.create(model=cfg["deployment"], input=data)

        if isinstance(data, list):
            return [e.embedding for e in result.data]
        return result.data[0].embedding

    def invoke_gpt_35(
        self,
        input_text: str,
        temperature: float = 0,
        top_p: float = 0,
    ) -> dict:
        """
        Generate a response using GPT-3.5.

        Args:
            input_text: User message to send
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter

        Returns:
            Full API response as dictionary
        """
        cfg = self.config.get_gpt_35_config()

        client = AzureOpenAI(
            azure_endpoint=cfg["api_url"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
        )

        resp = client.chat.completions.create(
            model=cfg["deployment"],
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ],
        )
        return resp.model_dump()

    def invoke_gpt_4o(
        self,
        input_data: Optional[str] = None,
        path_to_file: Optional[str] = None,
        temperature: float = 0,
        top_p: float = 0,
        response_type: Optional[str] = None,
        schema=None,
    ) -> dict:
        """
        Generate a response using GPT-4o, optionally with image/PDF input.

        Args:
            input_data: Text input or None for file-only analysis
            path_to_file: Path to image or PDF file to analyze
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            response_type: None, "json", or "structured"
            schema: Schema for structured output

        Returns:
            Full API response as dictionary
        """
        cfg = self.config.get_gpt_4o_config()

        if cfg["resource_flag"] == "KEY":
            client = AzureOpenAI(
                azure_endpoint=cfg["api_url"],
                api_key=cfg["api_key"],
                api_version=cfg["api_version"],
            )
        else:
            client = AzureOpenAI(
                azure_endpoint=cfg["api_url"],
                azure_ad_token_provider=self.config.get_openai_token,
                api_version=cfg["api_version"],
                default_headers={"projectId": cfg["project_id"]},
            )

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        if isinstance(input_data, str) and not path_to_file:
            messages.append({"role": "user", "content": input_data})
        else:
            base64_images = []
            if path_to_file:
                ext = os.path.splitext(path_to_file)[1].lower()
                if ext == ".pdf":
                    with open(path_to_file, "rb") as f:
                        base64_images += _convert_pdf_to_base64_images(f.read())
                else:
                    with open(path_to_file, "rb") as f:
                        base64_images.append(base64.b64encode(f.read()).decode("utf-8"))

            user_content = [
                {"type": "text", "text": input_data or "Analyze the provided document."},
            ]
            for b64_img in base64_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                })

            messages.append({"role": "user", "content": user_content})

        response_format = {"type": "text"}
        if response_type == "json":
            response_format = {"type": "json_object"}
        elif response_type == "structured":
            response_format = schema

        resp = client.beta.chat.completions.parse(
            model=cfg["deployment"],
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            response_format=response_format,
        )
        return resp.model_dump()

    async def realtime_voice(
        self,
        *,
        voice: str = "alloy",
        server_vad: bool = True,
        instructions: Optional[str] = None,
        input_audio_transcription: Optional[dict] = None,
    ):
        """
        Connect to the Realtime Voice API.

        Args:
            voice: Voice to use (alloy, echo, shimmer, etc.)
            server_vad: Use server-side voice activity detection
            instructions: System instructions for the assistant
            input_audio_transcription: Config for transcribing user audio

        Returns:
            RealtimeConnection for sending/receiving audio

        Example:
            async with await client.realtime_voice() as conn:
                # Send audio
                await conn.input_audio_append(pcm_bytes)

                # Receive responses
                async for event in conn.iter_events():
                    print(event)
        """
        rtc = RealtimeVoiceClient(self.config, ssl_verify=True)

        session_params = RealtimeSessionParams(
            voice=voice,
            turn_detection={"type": "server_vad"} if server_vad else None,
            input_audio_transcription=input_audio_transcription,
        )

        response_params = RealtimeResponseParams(
            modalities=["text", "audio"],
            instructions=instructions,
        )

        return await rtc.connect(session_params=session_params, response_params=response_params)
