from typing import AsyncIterator, Tuple
from wiz_tts.tts_adapters.openai_tts import OpenAITTS
from wiz_tts.tts_adapters.groq_tts import GroqTTS

GROQ_VOICES = [
    "Arista-PlayAI",
    "Atlas-PlayAI",
    "Basil-PlayAI",
    "Briggs-PlayAI",
    "Calum-PlayAI",
    "Celeste-PlayAI",
    "Cheyenne-PlayAI",
    "Chip-PlayAI",
    "Cillian-PlayAI",
    "Deedee-PlayAI",
    "Fritz-PlayAI",
    "Gail-PlayAI",
    "Indigo-PlayAI",
    "Mamaw-PlayAI",
    "Mason-PlayAI",
    "Mikail-PlayAI",
    "Mitch-PlayAI",
    "Quinn-PlayAI",
    "Thunder-PlayAI",
]

OPENAI_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"
]

class TextToSpeech:
    """Handles text-to-speech generation by selecting the appropriate TTS adapter."""

    def __init__(self):
        self.openai_tts = OpenAITTS()
        self.groq_tts = GroqTTS()

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
            model: The TTS model to use

        Returns:
            Tuple of (sample_rate, AsyncIterator[bytes]) containing the audio sample rate
            and an async iterator of audio chunks
        """
        if voice in GROQ_VOICES:
            return self.groq_tts.sample_rate, self.groq_tts.generate_speech(text, voice, "playai-tts")
        else:
            return self.openai_tts.sample_rate, self.openai_tts.generate_speech(text, voice, instructions, model)
