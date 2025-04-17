from openai import AsyncOpenAI
from typing import AsyncIterator

class TextToSpeech:
    """Handles text-to-speech generation using OpenAI's API."""

    def __init__(self):
        self.client = AsyncOpenAI()

    async def generate_speech(
        self,
        text: str,
        voice: str = "coral",
        instructions: str = "",
        model: str = "tts-1"
    ) -> AsyncIterator[bytes]:
        """
        Generate speech from text using OpenAI's TTS API.

        Args:
            text: The text to convert to speech
            voice: The voice to use
            instructions: Voice style instructions
            model: The TTS model to use

        Returns:
            An async iterator of audio chunks
        """
        async with self.client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="pcm",
        ) as response:
            async for chunk in response.iter_bytes(1024):  # Use smaller chunks for lower latency
                yield chunk
