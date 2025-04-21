from typing import AsyncIterator
import asyncio
from groq import Groq

class GroqTTS:
    """Handles text-to-speech generation using Groq's API."""

    def __init__(self):
        self.client = Groq()
        self.sample_rate = 48000  # Groq returns 48kHz audio

    async def generate_speech(
        self,
        text: str,
        voice: str = "Thunder-PlayAI",
        model: str = "playai-tts"
    ) -> AsyncIterator[bytes]:
        """
        Generate speech from text using Groq's API.

        Args:
            text: The text to convert to speech
            voice: The voice to use
            model: The TTS model to use

        Returns:
            An async iterator of audio chunks
        """
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            response_format="wav",
            input=text,
        )

        # Convert synchronous iterator to asynchronous iterator
        for chunk in response.iter_bytes(1024):
            yield chunk
            # Small delay to allow other tasks to run
            await asyncio.sleep(0)
