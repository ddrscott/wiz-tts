import asyncio
import argparse
import sys
import signal
from typing import Optional

from rich.console import Console
from rich.status import Status

from wiz_tts.tts import TextToSpeech
from wiz_tts.audio import AudioPlayer

console = Console()
audio_player = None

def signal_handler(sig, frame):
    """Handle Ctrl+C by stopping audio playback."""
    global audio_player
    if audio_player:
        console.print("\n[bold red]Playback interrupted![/]")
        audio_player.stop()
    sys.exit(0)

async def async_main(text: str, voice: str = "coral", instructions: str = "", model: str = "tts-1") -> None:
    """Main function to handle TTS generation and playback."""
    global audio_player

    console.print(f"Generating speech with {model}, voice: {voice}...")

    # Initialize services
    tts = TextToSpeech()
    audio_player = AudioPlayer()
    audio_player.start()

    try:
        with console.status("Playing...") as status:
            async for chunk in tts.generate_speech(text, voice, instructions, model):
                # Process chunk and get visualization data
                viz_data = audio_player.play_chunk(chunk)

                # Update display if visualization data is available
                if viz_data:
                    status.update(f"[{viz_data['counter']}] ▶ {viz_data['histogram']}")

    finally:
        # Ensure we always clean up
        audio_player.stop()
        console.print("Playback complete!")

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

    try:
        asyncio.run(async_main(text, args.voice, args.instructions, args.model))
    except KeyboardInterrupt:
        # This is a fallback in case the signal handler doesn't work
        console.print("\n[bold]Playback cancelled[/]")
        if audio_player:
            audio_player.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
