import asyncio
import argparse
import numpy as np
import sys
from scipy.fftpack import fft
import sounddevice as sd
from typing import Any, Optional, List

from rich.console import Console
from rich.status import Status

from openai import AsyncOpenAI

console = Console()
openai = AsyncOpenAI()

# Constants for audio processing
SAMPLE_RATE = 24000  # OpenAI's PCM format is 24kHz
CHUNK_SIZE = 4800  # 200ms chunks for visualization (5 updates per second)

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

async def async_main(text: str, voice: str = "coral", instructions: str = "", model: str = "tts-1") -> None:
    """Main function to handle TTS generation and playback."""

    console.print(f"Generating speech with {model}, voice: {voice}...")

    # Setup the output stream before getting data
    stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
    )

    stream.start()

    # Buffer to store small chunks for FFT analysis
    audio_buffer = []
    chunk_counter = 0

    with console.status("Playing...") as status:
        async with openai.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="pcm",
        ) as response:
            # Stream chunks directly to audio output
            async for chunk in response.iter_bytes(1024):  # Use smaller chunks for lower latency
                # Write directly to sound device
                stream.write(chunk)

                # Store for visualization
                chunk_data = np.frombuffer(chunk, dtype=np.int16)
                audio_buffer.extend(chunk_data)

                # When we have enough data for FFT analysis
                if len(audio_buffer) >= CHUNK_SIZE:
                    # Calculate FFT on current chunk
                    fft_result = fft(audio_buffer[:CHUNK_SIZE])
                    histogram = generate_histogram(fft_result)

                    # Update display
                    chunk_counter += 1
                    status.update(f"[{chunk_counter}] ▶ {histogram}")

                    # Keep only the newest data
                    audio_buffer = audio_buffer[CHUNK_SIZE:]

    # Close the stream after playback finishes
    stream.stop()
    stream.close()
    console.print("Playback complete!")

def read_stdin_text():
    """Read text from stdin if available."""
    # Check if stdin has data
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None

def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Convert text to speech with visualization")
    parser.add_argument("text", nargs="?", default=None,
                        help="Text to convert to speech (default: reads from stdin or uses a sample text)")
    parser.add_argument("--voice", "-v", default="coral",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral"],
                        help="Voice to use for speech (default: coral)")
    parser.add_argument("--instructions", "-i", default="",
                        help="Instructions for the speech style")
    parser.add_argument("--model", "-m", default="tts-1",
                        choices=["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
                        help="TTS model to use (default: tts-1)")

    args = parser.parse_args()

    # First priority: command line argument
    # Second priority: stdin
    # Third priority: default text
    text = args.text
    if text is None:
        text = read_stdin_text()
    if text is None:
        text = "Today is a wonderful day to build something people love!"

    asyncio.run(async_main(text, args.voice, args.instructions, args.model))

if __name__ == "__main__":
    main()