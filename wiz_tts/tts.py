from typing import AsyncIterator, Tuple, Dict, Optional, Any
import importlib
import json
from pathlib import Path
import importlib.resources
import os

class TextToSpeech:
    """Handles text-to-speech generation by selecting the appropriate TTS adapter."""

    def __init__(self):
        # Load voice configuration from JSON file
        self.voice_adapters, self.model_overrides = self._load_voice_configuration()

    def _load_voice_configuration(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Load voice configuration from the voices.json file.

        Returns:
            Tuple of:
              - Dict mapping voice names to adapter module names
              - Dict mapping adapter names to model overrides
        """
        voice_mapping = {}
        model_overrides = {}

        # Try multiple possible locations for voices.json
        possible_paths = [
            # Development location (project root)
            Path(__file__).parent.parent / "voices.json",
            # Installed package location (inside package)
            Path(__file__).parent / "voices.json",
            # Alternative installed package location
            Path(__file__).parent / "data" / "voices.json"
        ]

        config_data = None
        for voices_path in possible_paths:
            try:
                if voices_path.exists():
                    with open(voices_path, "r") as f:
                        config_data = json.load(f)
                        break  # Stop once we've found and loaded the file
            except Exception as e:
                print(f"Error loading {voices_path}: {e}")
                continue  # Try the next path

        if config_data:
            # Process the configuration
            for config in config_data:
                adapter_name = config["adapter"]

                # Store model override if specified
                if "override-model" in config:
                    model_overrides[adapter_name] = config["override-model"]

                # Map all voices to this adapter
                voices = config["voices"]
                for voice in voices:
                    voice_mapping[voice] = adapter_name
        else:
            raise FileNotFoundError("No valid voices.json file found in expected locations.")

        return voice_mapping, model_overrides

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
