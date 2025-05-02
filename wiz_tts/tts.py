from typing import AsyncIterator, Tuple
import importlib
import asyncio
from wiz_tts.voices import VOICE_ADAPTERS, MODEL_OVERRIDES

class TextToSpeech:
    """Handles text-to-speech generation by selecting the appropriate TTS adapter."""

    def __init__(self):
        # Voice configurations are loaded directly from the voices module
        self.voice_adapters = VOICE_ADAPTERS
        self.model_overrides = MODEL_OVERRIDES
        
    def get_event_loop(self):
        """Get or create an event loop for async operations."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def generate_speech(
        self,
        text: str,
        voice: str = "coral",
        instructions: str = "",
        model: str = "tts-1"
    ) -> Tuple[int, AsyncIterator[bytes]]:
        """
        Generate speech from text using the appropriate TTS adapter based on voice.

        Args:
            text: The text to convert to speech
            voice: The voice to use
            instructions: Voice style instructions (only supported by OpenAI)
            model: The TTS model to use (may be overridden by configuration)

        Returns:
            Tuple of (sample_rate, AsyncIterator[bytes]) containing the audio sample rate
            and an async iterator of audio chunks

        Raises:
            ValueError: If the voice is not recognized
        """
        # Determine which adapter to use
        adapter_name = self.voice_adapters.get(voice)
        if not adapter_name:
            available_voices = ", ".join(sorted(self.voice_adapters.keys()))
            raise ValueError(f"Unknown voice: {voice}. Available voices: {available_voices}")

        # Dynamically import the appropriate adapter
        adapter = importlib.import_module(f"wiz_tts.tts_adapters.{adapter_name}")

        # Check if there's a model override for this adapter
        actual_model = self.model_overrides.get(adapter_name, model)

        # Forward to the appropriate adapter
        return adapter.SAMPLE_RATE, adapter.generate_speech(text, voice, instructions, actual_model)
