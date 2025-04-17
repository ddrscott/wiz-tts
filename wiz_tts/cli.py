import asyncio
import argparse
import numpy as np
from scipy.fftpack import fft
import sounddevice as sd
from typing import Any

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

    console.print(f"generating...")
    async with openai.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        with console.status("playing...") as status:
            # First collect the full audio data
            audio_buffer = bytearray()
            async for chunk in response.iter_bytes():
                audio_buffer.extend(chunk)

            # Convert to numpy array - PCM is 16-bit signed integers
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

            # Normalize to float32 in range [-1, 1] for sounddevice
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Reshape for mono (1 channel)
            audio_content = audio_float.reshape(-1, 1)

            # Set up the sound device stream for better playback
            status.update("Playing audio...")

            # Use the streaming approach with callback for smooth playback
            loop = asyncio.get_event_loop()
            event = asyncio.Event()
            position = 0
            total_frames = len(audio_content)

            def callback(outdata: np.ndarray, frame_count: int,
                        _time_info: Any, _status: Any) -> None:
                nonlocal position

                if position >= total_frames:
                    loop.call_soon_threadsafe(event.set)
                    raise sd.CallbackStop

                # Calculate how many frames to copy
                chunk_size = min(frame_count, total_frames - position)
                outdata[:chunk_size] = audio_content[position:position + chunk_size]
                outdata[chunk_size:] = 0  # Pad with silence if needed

                # Create visual representation of current chunk
                current_chunk = audio_content[position:position + chunk_size, 0]
                fft_result = fft(current_chunk)
                histogram = generate_histogram(fft_result)

                # Update progress display
                progress_pct = min(position / total_frames * 100, 100)
                progress = f"[{int(progress_pct)}%]"
                loop.call_soon_threadsafe(lambda: status.update(f"{progress} ▶ {histogram}"))

                # Move position forward
                position += chunk_size

            # Start playback with the callback
            stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=callback,
                dtype=np.float32
            )

            with stream:
                await event.wait()

def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Convert text to speech with visualization")
    parser.add_argument("text", nargs="?", default="Today is a wonderful day to build something people love!",
                        help="Text to convert to speech (default: \"Today is a wonderful day to build something people love!\")")
    parser.add_argument("--voice", "-v", default="coral",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral"],
                        help="Voice to use for speech (default: coral)")
    parser.add_argument("--instructions", "-i", default="",
                        help="Instructions for the speech style")
    parser.add_argument("--model", "-m", default="tts-1",
                        choices=["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
                        help="TTS model to use (default: tts-1)")

    args = parser.parse_args()

    asyncio.run(async_main(args.text, args.voice, args.instructions, args.model))

if __name__ == "__main__":
    main()
