import argparse
import sys
import signal
import os
import time
import re
import queue
import threading
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

from rich.console import Console
from rich.status import Status
from rich.table import Table

from wiz_tts.tts import TextToSpeech
from wiz_tts.audio import AudioPlayer
from wiz_tts.voices import VOICE_ADAPTERS, GROQ_VOICES, OPENAI_VOICES, MODEL_OVERRIDES

console = Console()
audio_player = None

def signal_handler(sig, frame):
    """Handle Ctrl+C by stopping audio playback and saving audio."""
    global audio_player
    if audio_player:
        console.print("\n[bold red]Playback interrupted![/]")
        # Don't call sys.exit() immediately - allow for cleanup
        audio_player.stop()

        # If we have a data directory, make sure to finalize the audio
        if audio_player.data_dir:
            saved_path = audio_player.save_audio()
            if saved_path:
                console.print(f"[green]Audio saved to:[/] {saved_path}")

    sys.exit(0)

def split_text(text: str, split_method: str) -> List[str]:
    """Split text into smaller segments based on the specified method."""
    if split_method == "period":
        # Split by sentences, trying to be smart about abbreviations
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    elif split_method == "paragraph":
        # Split by blank lines or new lines
        paragraphs = [p.strip() for p in text.split('\n\n')]
        if len(paragraphs) == 1:  # If no double line breaks, try single line breaks
            paragraphs = [p.strip() for p in text.split('\n')]
        return [p for p in paragraphs if p]

    # Default: return the entire text as a single segment
    return [text]

def process_tts(text: str, voice: str = "coral", instructions: str = "", model: str = "tts-1",
         data_dir: Optional[str] = None, bitrate: str = "24k", split_method: Optional[str] = None) -> None:
    """Main function to handle TTS generation and playback."""
    global audio_player

    # Determine the actual model that will be used based on voice and model overrides
    adapter_name = VOICE_ADAPTERS.get(voice)
    actual_model = MODEL_OVERRIDES.get(adapter_name, model) if adapter_name else model

    console.print(f"wiz-tts with model: {actual_model}, voice: {voice}")

    # Prepare data directory if provided
    data_path = Path(data_dir) if data_dir else None

    # Initialize text-to-speech service
    tts = TextToSpeech()

    # Split text if requested
    if split_method:
        segments = split_text(text, split_method)
        console.print(f"Split text into {len(segments)} segments")
    else:
        segments = [text]

    # Create communication queues
    audio_queue = queue.Queue(maxsize=100)     # Queue for audio chunks
    status_queue = queue.Queue()               # Queue for status updates
    sample_rate_queue = queue.Queue()          # Queue for sample rate
    completion_event = threading.Event()       # Event to signal overall completion
    playback_done_event = threading.Event()    # Event to signal playback completion
    save_done_event = threading.Event()        # Event to signal save completion

    # Create thread-safe lock for console access
    console_lock = threading.Lock()

    # Keep track of path announcements
    file_path_announced = False

    # Thread for audio generation
    def generate_audio_thread():
        try:
            # Handle first segment special case - need to determine sample rate
            first_segment = segments[0]
            status_queue.put(f"Generating segment 1/{len(segments)}...")

            # For the first segment, we need the actual sample rate
            sample_rate, speech_gen = tts.generate_speech(first_segment, voice, instructions, model)
            sample_rate_queue.put(sample_rate)

            # Process the first segment's audio generator
            segment_buffer = bytearray()

            # Process the async generator in a loop
            loop = tts.get_event_loop()

            async def collect_chunks(generator):
                buffer = bytearray()
                async for chunk in generator:
                    buffer.extend(chunk)
                return bytes(buffer)

            # Get all audio for the first segment
            first_segment_audio = loop.run_until_complete(collect_chunks(speech_gen))

            # Put first segment in queue with proper chunk size
            chunk_size = 4096
            for i in range(0, len(first_segment_audio), chunk_size):
                chunk = first_segment_audio[i:i+chunk_size]
                audio_queue.put(chunk)

            # Process remaining segments
            for i, segment in enumerate(segments[1:], start=2):
                status_queue.put(f"Generating segment {i}/{len(segments)}...")

                # Generate silence between segments if needed
                if split_method:
                    # Use very minimal pauses for natural flow
                    pause_duration = 0.3 if split_method == "paragraph" else 0.15

                    # Create a silence buffer with a smooth fade-in and fade-out to prevent crackling
                    silence_length = int(sample_rate * pause_duration)
                    fade_samples = int(sample_rate * 0.02)  # 20ms fade

                    # Create silence with two bytes per sample (16-bit audio)
                    silence_buffer = bytearray(silence_length * 2)

                    # Add a small amount of quiet audio instead of complete silence
                    # This helps prevent sound card clicks
                    for i in range(silence_length):
                        value = 2  # Very small non-zero value to prevent clicks

                        # Apply fade in/out
                        if i < fade_samples:
                            # Fade in
                            factor = i / fade_samples
                            value = int(value * factor)
                        elif i > silence_length - fade_samples:
                            # Fade out
                            factor = (silence_length - i) / fade_samples
                            value = int(value * factor)

                        # Set the bytes (little endian)
                        silence_buffer[i*2] = value & 0xFF
                        silence_buffer[i*2+1] = (value >> 8) & 0xFF

                    audio_queue.put(bytes(silence_buffer))

                # Generate the segment
                _, speech_gen = tts.generate_speech(segment, voice, instructions, model)

                # Get all audio for this segment
                segment_audio = loop.run_until_complete(collect_chunks(speech_gen))

                # Put segment in queue with proper chunk size
                for i in range(0, len(segment_audio), chunk_size):
                    chunk = segment_audio[i:i+chunk_size]
                    audio_queue.put(chunk)

                status_queue.put(f"Queued segment {i}/{len(segments)}")

            # Signal end of audio data
            audio_queue.put(None)

        except Exception as e:
            with console_lock:
                console.print(f"[bold red]Error in audio generation: {str(e)}[/]")
            # Signal end in case of error
            audio_queue.put(None)

    # Thread for audio playback
    def playback_thread():
        nonlocal file_path_announced

        try:
            # Wait for the sample rate from the generator thread
            sample_rate = sample_rate_queue.get()

            # Initialize audio player
            global audio_player
            audio_player = AudioPlayer(data_path)

            # Set metadata if we're saving
            if data_path:
                metadata = {
                    "text": text,
                    "voice": voice,
                    "model": actual_model,
                    "instructions": instructions,
                    "timestamp": time.time(),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "bitrate": bitrate,
                    "split_method": split_method,
                    "sample_rate": sample_rate
                }
                audio_player.set_metadata(metadata)

            # Start the audio player
            audio_player.start(sample_rate)

            # Play audio chunks
            status_queue.put("Playing audio...")

            # Track visualization update timing
            last_viz_time = time.time()
            viz_interval = 0.1  # Update visualization every 100ms max

            chunk = None
            while True:
                try:
                    # Get next chunk, with timeout
                    chunk = audio_queue.get(timeout=1.0)

                    # Check for end of audio
                    if chunk is None:
                        break

                    # Play the chunk
                    viz_data = audio_player.play_chunk(chunk)

                    # Update visualization if needed
                    current_time = time.time()
                    if viz_data and current_time - last_viz_time >= viz_interval:
                        status_queue.put({
                            "counter": viz_data.get("counter", 0),
                            "histogram": viz_data.get("histogram", ""),
                            "saving_to": viz_data.get("saving_to", None)
                        })

                        # Check if this is the first chunk with file path info
                        if "saving_to" in viz_data and not file_path_announced:
                            with console_lock:
                                console.print(f"[green]Saving audio to:[/] {viz_data['saving_to']}")
                            file_path_announced = True

                        last_viz_time = current_time

                    # Calculate sleep time based on audio duration for smooth playback
                    sleep_time = len(chunk) / (sample_rate * 2) * 0.8  # *2 for 16-bit samples, 0.8 to prevent underruns
                    time.sleep(sleep_time)

                except queue.Empty:
                    # If queue is empty but we haven't gotten None yet, continue waiting
                    continue

            # Play a bit of silence at the end to ensure complete playback
            status_queue.put("Finalizing playback...")

            # Play silence to flush audio system
            silence = bytes(int(sample_rate * 0.5))  # 0.5 second of silence
            audio_player.play_chunk(silence)
            time.sleep(0.5)

            # Signal playback completion
            playback_done_event.set()
            status_queue.put("Playback completed")

        except Exception as e:
            with console_lock:
                console.print(f"[bold red]Error in audio playback: {str(e)}[/]")
            # Signal completion in case of error
            playback_done_event.set()

    # Thread for saving audio file
    def save_audio_thread():
        try:
            # Wait for playback to finish first
            playback_done_event.wait()

            # Only save if we have a data directory
            if data_path and audio_player:
                status_queue.put("Saving audio file...")

                # Save the audio
                saved_path = audio_player.save_audio()

                # Announce completion if needed
                if saved_path and not file_path_announced:
                    with console_lock:
                        console.print(f"[green]Audio saved to:[/] {saved_path}")
                elif saved_path:
                    with console_lock:
                        console.print(f"[green]Audio finalized.[/]")
        except Exception as e:
            with console_lock:
                console.print(f"[bold red]Error saving audio: {str(e)}[/]")
        finally:
            # Always signal completion
            save_done_event.set()

    # Thread for status updates
    def status_update_thread():
        try:
            with console.status("Initializing...") as status:
                while not completion_event.is_set():
                    try:
                        # Get status update with timeout
                        update = status_queue.get(timeout=0.1)

                        # Handle different types of updates
                        if isinstance(update, dict):
                            # Visualization update
                            status.update(f"[{update['counter']}] ▶ {update['histogram']}")
                        else:
                            # Text status update
                            status.update(update)

                    except queue.Empty:
                        # No update available, continue
                        pass
        except Exception as e:
            with console_lock:
                console.print(f"[bold red]Error in status updates: {str(e)}[/]")

    # Start all threads
    generation_thread = threading.Thread(target=generate_audio_thread, daemon=True)
    playback_thread = threading.Thread(target=playback_thread, daemon=True)
    save_thread = threading.Thread(target=save_audio_thread, daemon=True)
    status_thread = threading.Thread(target=status_update_thread, daemon=True)

    generation_thread.start()
    playback_thread.start()
    save_thread.start()
    status_thread.start()

    try:
        # Wait for all processing to complete
        save_done_event.wait()

        # Signal overall completion
        completion_event.set()

        # Wait for threads to finish
        generation_thread.join(timeout=1.0)
        playback_thread.join(timeout=1.0)
        save_thread.join(timeout=1.0)
        status_thread.join(timeout=1.0)

    finally:
        # Clean up
        if audio_player:
            audio_player.stop()
        console.print("Playback complete!")

def show_voices():
    """Display a list of configured voices and required environment variables."""
    console.print("[bold]Available TTS Voices[/]\n")

    # Create a table for Groq voices
    groq_table = Table(title="Groq Voices (PlayAI-TTS)")
    groq_table.add_column("Voice", style="cyan")

    # Add Groq voices to the table
    groq_voices = sorted([v for v in GROQ_VOICES])
    for voice in groq_voices:
        groq_table.add_row(voice)

    # Create a table for OpenAI voices
    openai_table = Table(title="OpenAI Voices")
    openai_table.add_column("Voice", style="green")
    openai_table.add_column("Model", style="yellow")

    # Add OpenAI voices to the table
    openai_voices = sorted([v for v in OPENAI_VOICES])
    for voice in openai_voices:
        openai_table.add_row(voice, "tts-1 / tts-1-hd / gpt-4o-mini-tts")

    # Print the tables
    console.print(groq_table)
    console.print("\n")
    console.print(openai_table)

    # Display warnings about required environment variables
    console.print("\n[bold yellow]Required Environment Variables:[/]")

    # Check for OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        console.print("✅ [green]OPENAI_API_KEY is set[/] (required for OpenAI voices)")
    else:
        console.print("❌ [red]OPENAI_API_KEY is not set[/] (required for OpenAI voices)")

    # Check for Groq API key
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        console.print("✅ [green]GROQ_API_KEY is set[/] (required for Groq voices)")
    else:
        console.print("❌ [red]GROQ_API_KEY is not set[/] (required for Groq voices)")

    # Optional: Check for data directory
    data_dir = os.environ.get("WIZ_TTS_DATA_DIR")
    if data_dir:
        console.print(f"✅ [green]WIZ_TTS_DATA_DIR is set[/] (optional, for saving audio files): {data_dir}")
    else:
        console.print("ℹ️  [blue]WIZ_TTS_DATA_DIR is not set[/] (optional, for saving audio files)")

def read_stdin_text():
    """Read text from stdin if available."""
    # Check if stdin has data
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None

def main():
    """Entry point for the CLI."""
    # Register the signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert text to speech with visualization")

    # Add main arguments
    parser.add_argument("text", nargs="?", default=None,
                        help="Text to convert to speech (default: reads from stdin or uses a sample text)")
    parser.add_argument("--voice", "-v", default="ash",
                        help="Voice to use for speech (default: ash)")
    parser.add_argument("--instructions", "-i", default="",
                        help="Instructions for the speech style")
    parser.add_argument("--model", "-m", default="gpt-4o-mini-tts",
                        choices=["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
                        help="TTS model to use (default: gpt-4o-mini-tts)")
    parser.add_argument("--split", choices=["period", "paragraph"],
                        help="Split input text by periods or paragraphs and add natural pauses")

    # Get data directory from environment variable if set, otherwise None
    default_data_dir = os.environ.get("WIZ_TTS_DATA_DIR")
    parser.add_argument("--data-dir", "-d", default=default_data_dir,
                        help="Directory to save audio files and metadata (default: $WIZ_TTS_DATA_DIR if set)")
    parser.add_argument("--bitrate", "-b", default="24k",
                        help="Audio bitrate for saved files (default: 24k)")

    # Add voices command as a flag
    parser.add_argument("--voices", action="store_true",
                        help="List available voices and check environment variables")

    # Parse arguments
    args = parser.parse_args()

    # If no arguments provided at all, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Handle --voices flag
    if args.voices:
        # Just show the voices and exit
        show_voices()
        return

    # Default behavior: process the text argument for TTS
    text = args.text
    if text is None:
        text = read_stdin_text()
    if text is None:
        text = "Today is a wonderful day to build something people love!"

    try:
        process_tts(text, args.voice, args.instructions, args.model,
                   args.data_dir, args.bitrate, args.split)
    except KeyboardInterrupt:
        # This is a fallback in case the signal handler doesn't work
        console.print("\n[bold]Playback cancelled[/]")
        if audio_player:
            audio_player.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
