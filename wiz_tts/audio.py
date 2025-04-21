import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from pydub import AudioSegment
import os
import json
import time
from pathlib import Path
from typing import List, Iterator, AsyncIterator, Optional, Dict, Any, Union

# Default sample rate (used as fallback)
DEFAULT_SAMPLE_RATE = 24000  # OpenAI's PCM format is 24kHz
CHUNK_SIZE = 4800  # 200ms chunks for visualization (5 updates per second)

class AudioPlayer:
    """Handles audio playback and visualization processing."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.stream = None
        self.audio_buffer = []
        self.chunk_counter = 0
        self.is_playing = False
        self.data_dir = data_dir
        self.full_audio_data = bytearray()  # Store all audio data for saving
        self.metadata = {}  # Store metadata for inference options
        self.timestamp = int(time.time())
        self.metadata_saved = False
        self.audio_file = None  # Will hold the file handle for incremental saving
        self.sample_rate = DEFAULT_SAMPLE_RATE  # Default sample rate

    def start(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """Initialize and start the audio stream with the specified sample rate."""
        self.sample_rate = sample_rate

        self.stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
        )
        self.stream.start()
        # Play a brief silence to prevent initial crackling
        silence = bytes(1024 * 10)  # 1024 bytes of silence (all zeros)
        self.stream.write(silence)
        self.audio_buffer = []
        self.full_audio_data = bytearray()  # Reset the full audio data
        self.chunk_counter = 0
        self.is_playing = True

    def stop(self):
        """Stop and close the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata for the audio file and save it immediately."""
        self.metadata = metadata

        # Add the correct sample rate to metadata
        self.metadata["sample_rate"] = self.sample_rate

        # Save metadata immediately if we have a data directory
        if self.data_dir and not self.metadata_saved:
            # Ensure the data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Save metadata to JSON file
            metadata_path = self.data_dir / f"audio_{self.timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            self.metadata_saved = True

    def play_chunk(self, chunk: bytes) -> dict:
        """
        Play an audio chunk and process for visualization.

        Returns:
            dict: Visualization data including histogram and counter
        """
        if not self.is_playing or not self.stream:
            return None

        # Write directly to sound device
        self.stream.write(chunk)

        # If we're saving, store the full audio data and start saving WebM if first chunk
        if self.data_dir:
            self.full_audio_data.extend(chunk)

            # Start WebM file if this is the first audio chunk and we have metadata
            if len(self.full_audio_data) == len(chunk) and self.metadata:
                # Ensure the data directory exists
                self.data_dir.mkdir(parents=True, exist_ok=True)

                # Create WebM file path
                self.webm_path = self.data_dir / f"audio_{self.timestamp}.webm"

                # Return file path for progress indicator
                return {
                    "counter": 0,
                    "histogram": "",
                    "saving_to": str(self.webm_path)
                }

        # Store for visualization
        chunk_data = np.frombuffer(chunk, dtype=np.int16)
        self.audio_buffer.extend(chunk_data)

        # Calculate visualization size based on sample rate
        vis_chunk_size = int(self.sample_rate * 0.2)  # 200ms worth of samples

        # Process visualization data when we have enough
        if len(self.audio_buffer) >= vis_chunk_size:
            # Calculate FFT on current chunk
            fft_result = fft(self.audio_buffer[:vis_chunk_size])
            histogram = generate_histogram(fft_result)

            # Update counter
            self.chunk_counter += 1

            # Keep only the newest data
            self.audio_buffer = self.audio_buffer[vis_chunk_size:]

            return {
                "counter": self.chunk_counter,
                "histogram": histogram
            }

        return None

    def save_audio(self) -> Optional[Path]:
        """
        Finalize the collected audio data to a WebM file with metadata.
        The metadata file is already saved at the beginning.

        Returns:
            Path: The path to the saved file, or None if no data or data_dir
        """
        if not self.data_dir or not self.full_audio_data:
            return None

        # Use the same webm_path that was created at the beginning
        file_path = getattr(self, 'webm_path', None)

        # If we don't have a path yet (unlikely), create one
        if file_path is None:
            file_path = self.data_dir / f"audio_{self.timestamp}.webm"

        # Convert raw PCM to AudioSegment
        audio = AudioSegment(
            data=bytes(self.full_audio_data),
            sample_width=2,  # 16-bit audio (2 bytes)
            frame_rate=self.sample_rate,  # Use the actual sample rate
            channels=1
        )

        # Save as WebM format with Opus codec optimized for speech
        audio.export(
            str(file_path),
            format="webm",
            parameters=[
                # Use Opus codec (excellent for speech)
                "-c:a", "libopus",
                # Optimize for speech
                "-application", "voip",
                # Bitrate in kbps (using value from metadata or default)
                "-b:a", self.metadata.get("bitrate", "24k"),
                # Add metadata
                "-metadata", f"metadata={json.dumps(self.metadata)}"
            ]
        )

        return file_path

def generate_histogram(fft_values: np.ndarray, width: int = 12) -> str:
    """Generate a text-based histogram from FFT values."""
    # Use lower frequencies (more interesting for speech)
    fft_values = np.abs(fft_values[:len(fft_values)//4])

    # Group the FFT values into bins
    bins = np.array_split(fft_values, width)
    bin_means = [np.mean(bin) for bin in bins]

    # Normalize values
    max_val = max(bin_means) if any(bin_means) else 1.0
    # Handle potential NaN values by replacing them with 0.0
    normalized = [min(val / max_val, 1.0) if not np.isnan(val) else 0.0 for val in bin_means]

    # Create histogram bars using Unicode block characters
    bars = ""
    for val in normalized:
        # Check for NaN values before converting to int
        if np.isnan(val):
            height = 0
        else:
            height = int(val * 8)  # 8 possible heights with Unicode blocks

        if height == 0:
            bars += " "
        else:
            # Unicode block elements from 1/8 to full block
            blocks = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            bars += blocks[height]

    return bars
