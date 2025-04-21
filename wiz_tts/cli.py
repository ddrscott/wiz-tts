import asyncio
import argparse
import sys
import signal
import os
import time
from pathlib import Path
from typing import Optional, List, Dict

from rich.console import Console
from rich.status import Status
from rich.table import Table

from wiz_tts.tts import TextToSpeech
from wiz_tts.audio import AudioPlayer
from wiz_tts.voices import VOICE_ADAPTERS, GROQ_VOICES, OPENAI_VOICES

console = Console()
audio_player = None

def signal_handler(sig, frame):
    """Handle Ctrl+C by stopping audio playback."""
    global audio_player
    if audio_player:
        console.print("\n[bold red]Playback interrupted![/]")
        audio_player.stop()
    sys.exit(0)

async def async_main(text: str, voice: str = "coral", instructions: str = "", model: str = "tts-1", data_dir: Optional[str] = None, bitrate: str = "24k") -> None:
    """Main function to handle TTS generation and playback."""
    global audio_player

    console.print(f"wiz-tts with model: {model}, voice: {voice}")

    # Prepare data directory if provided
    data_path = Path(data_dir) if data_dir else None

    # Initialize services
    tts = TextToSpeech()

    # Initialize audio player (will configure sample rate during playback)
    audio_player = AudioPlayer(data_path)

    # Set metadata if we're saving
    if data_path:
        metadata = {
            "text": text,
            "voice": voice,
            "model": model,
            "instructions": instructions,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bitrate": bitrate
        }
        audio_player.set_metadata(metadata)

    file_path_announced = False
    try:
        with console.status("Generating...", refresh_per_second=60) as status:
            # Get speech data with sample rate
            sample_rate, speech_generator = tts.generate_speech(text, voice, instructions, model)

            # Configure audio player with the correct sample rate
            audio_player.start(sample_rate)

            async for chunk in speech_generator:
                # Process chunk and get visualization data
                viz_data = audio_player.play_chunk(chunk)

                # Update display if visualization data is available
                if viz_data:
                    # If this is the first chunk and we have a file path, announce it
                    if 'saving_to' in viz_data and not file_path_announced:
                        console.print(f"[green]Saving audio to:[/] {viz_data['saving_to']}")
                        file_path_announced = True
                        status.update(f"Generating... ▶ {viz_data.get('histogram', '')}")
                    else:
                        status.update(f"[{viz_data['counter']}] ▶ {viz_data['histogram']}")

    finally:
        # Finalize audio if requested
        if data_path:
            saved_path = audio_player.save_audio()
            if saved_path and not file_path_announced:
                console.print(f"[green]Audio saved to:[/] {saved_path}")
            elif saved_path:
                console.print(f"[green]Audio finalized.[/]")

        # Ensure we always clean up
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
        asyncio.run(async_main(text, args.voice, args.instructions, args.model, args.data_dir, args.bitrate))
    except KeyboardInterrupt:
        # This is a fallback in case the signal handler doesn't work
        console.print("\n[bold]Playback cancelled[/]")
        if audio_player:
            audio_player.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()