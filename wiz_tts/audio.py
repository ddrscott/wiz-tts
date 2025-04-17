import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from typing import List, Iterator, AsyncIterator

# Constants for audio processing
SAMPLE_RATE = 24000  # OpenAI's PCM format is 24kHz
CHUNK_SIZE = 4800  # 200ms chunks for visualization (5 updates per second)

class AudioPlayer:
    """Handles audio playback and visualization processing."""

    def __init__(self):
        self.stream = None
        self.audio_buffer = []
        self.chunk_counter = 0
        self.is_playing = False

    def start(self):
        """Initialize and start the audio stream."""
        self.stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
        )
        self.stream.start()
        self.audio_buffer = []
        self.chunk_counter = 0
        self.is_playing = True

    def stop(self):
        """Stop and close the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

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

        # Store for visualization
        chunk_data = np.frombuffer(chunk, dtype=np.int16)
        self.audio_buffer.extend(chunk_data)

        # Process visualization data when we have enough
        if len(self.audio_buffer) >= CHUNK_SIZE:
            # Calculate FFT on current chunk
            fft_result = fft(self.audio_buffer[:CHUNK_SIZE])
            histogram = generate_histogram(fft_result)

            # Update counter
            self.chunk_counter += 1

            # Keep only the newest data
            self.audio_buffer = self.audio_buffer[CHUNK_SIZE:]

            return {
                "counter": self.chunk_counter,
                "histogram": histogram
            }

        return None

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
