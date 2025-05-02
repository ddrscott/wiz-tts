import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from pydub import AudioSegment
import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Iterator, AsyncIterator, Optional, Dict, Any, Union

# Default sample rate (used as fallback)
DEFAULT_SAMPLE_RATE = 24000  # OpenAI's PCM format is 24kHz
CHUNK_SIZE = 4800  # 200ms chunks for visualization (5 updates per second)

class AudioPlayer:
    """Handles audio playback and visualization processing."""

    def __init__(self, data_dir: Optional[Path] = None, prebuffer_size: int = 5):
        self.stream = None
        self.audio_buffer = []
        self.playback_buffer = bytearray()
        self.prebuffer_size = prebuffer_size  # Number of chunks to buffer before playback
        self.buffered_chunks = 0
        self.chunk_counter = 0
        self.is_playing = False
        self.playback_complete = False
        self.has_received_all_chunks = False
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

        # Use a higher latency setting for more stable playback
        # This helps reduce crackling, especially at the start and end
        self.stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            blocksize=2048,  # Smaller block size for more consistent playback
            latency='high',   # Higher latency reduces crackling
            callback=None,    # No callback for direct write mode
            device=None,      # Use default output device
        )
        self.stream.start()

        # Play substantial silence to prevent initial crackling
        # This is critical for properly initializing the audio hardware
        silence_duration_ms = 300  # 300ms of silence
        silence_bytes = int(sample_rate * silence_duration_ms / 1000) * 2  # 16-bit samples = 2 bytes per sample
        silence = bytes(silence_bytes)

        # Write the silence in smaller chunks for smoother startup
        chunk_size = 1024
        for i in range(0, len(silence), chunk_size):
            self.stream.write(silence[i:i+chunk_size])
            time.sleep(0.001)  # Very small delay between chunks

        # Reset state
        self.audio_buffer = []
        self.playback_buffer = bytearray()
        self.buffered_chunks = 0
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
        self.playback_complete = True

    def mark_all_chunks_received(self):
        """Mark that all audio chunks have been received from the TTS API."""
        self.has_received_all_chunks = True

    async def drain_playback_buffer(self):
        """
        Mark playback as complete since we're using a different buffering strategy now.
        This is kept for compatibility with the existing code structure.
        """
        # With our new queue-based approach, the playback worker handles draining
        # We just need to mark completion
        self.playback_complete = True
        await asyncio.sleep(0.05)  # Small sleep just to ensure async context switch

    async def wait_for_playback_complete(self, timeout=60.0):
        """Wait until playback is complete or timeout is reached."""
        start_time = time.time()
        while not self.playback_complete and (time.time() - start_time) < timeout:
            # If we've received all chunks and the buffer is empty, playback is complete
            if self.has_received_all_chunks and len(self.playback_buffer) == 0:
                self.playback_complete = True
                break
            await asyncio.sleep(0.1)  # Wait a bit before checking again

        return self.playback_complete

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
        Process an audio chunk for playback and visualization.
        Since we're using a queue-based approach in the CLI, this now handles:
        1. Direct playback of the chunk
        2. Storing for saving to file
        3. Visualization processing

        Returns:
            dict: Visualization data including histogram and counter
        """
        if not self.is_playing or not self.stream:
            return None

        # If we're saving, store the full audio data
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

        # Play the chunk directly - playback is now managed by the CLI's buffering
        if self.stream and chunk:
            self.stream.write(chunk)

        # Process visualization
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

        try:
            # Stop the stream before saving to avoid any interference
            # We'll restart if needed
            streaming_active = False
            if self.stream and self.is_playing:
                streaming_active = True
                self.stream.stop()

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

        finally:
            # No need to restart the stream as we're typically done by now
            pass

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
