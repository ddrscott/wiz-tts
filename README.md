# Wiz TTS

A simple command-line tool for text-to-speech using OpenAI's API, featuring real-time FFT visualization.

## Installation

```bash
uv tool install -U wiz-tts

# or if you prefer pip
pip install wiz-tts
```

## Usage

After installation, you can run the tool with:

```bash
# Recommended: run with uv for best performance
uv run -- wiz-tts "Your text to convert to speech"

# Alternatively, run directly
wiz-tts "Your text to convert to speech"
```

Or pipe text from another command:

```bash
echo "Your text" | uv run -- wiz-tts
cat file.txt | uv run -- wiz-tts
```

### Options

```
usage: wiz-tts [-h] [--voice {alloy,echo,fable,onyx,nova,shimmer,coral}] [--instructions INSTRUCTIONS] 
               [--model {tts-1,tts-1-hd,gpt-4o-mini-tts}] [--data-dir DATA_DIR] [text]

Convert text to speech with visualization

positional arguments:
  text                  Text to convert to speech (default: reads from stdin or uses a sample text)

options:
  -h, --help            show this help message and exit
  --voice {alloy,echo,fable,onyx,nova,shimmer,coral}, -v {alloy,echo,fable,onyx,nova,shimmer,coral}
                        Voice to use for speech (default: coral)
  --instructions INSTRUCTIONS, -i INSTRUCTIONS
                        Instructions for the speech style
  --model {tts-1,tts-1-hd,gpt-4o-mini-tts}, -m {tts-1,tts-1-hd,gpt-4o-mini-tts}
                        TTS model to use (default: tts-1)
  --data-dir DATA_DIR, -d DATA_DIR
                        Directory to save audio files and metadata (default: $WIZ_TTS_DATA_DIR if set)
  --bitrate BITRATE, -b BITRATE
                        Audio bitrate for saved files (default: 24k)
```

### Examples

Basic usage:
```bash
uv run -- wiz-tts "Hello, world!"
```

Using stdin:
```bash
echo "Hello from stdin" | uv run -- wiz-tts
```

Using a different voice:
```bash
uv run -- wiz-tts --voice nova "Welcome to the future of text to speech!"
```

Adding speech instructions:
```bash
uv run -- wiz-tts --voice shimmer --instructions "Speak slowly and clearly" "This is important information."
```

Using a different model:
```bash
uv run -- wiz-tts --model tts-1-hd "This will be rendered in high definition."
```

Processing a text file:
```bash
cat story.txt | uv run -- wiz-tts --voice echo
```

Saving audio to a directory:
```bash
uv run -- wiz-tts "Save this speech to a file" --data-dir ./saved_audio
```

Using environment variable for audio saving:
```bash
# Set the environment variable
export WIZ_TTS_DATA_DIR=./saved_speeches

# Run without --data-dir, audio will be saved to ./saved_speeches
uv run -- wiz-tts "This will be saved using the environment variable"
```

Saving with custom audio compression:
```bash
# Higher bitrate for better quality but larger file size
uv run -- wiz-tts "High quality audio" --data-dir ./saved_audio --bitrate 64k

# Lower bitrate for smaller file size
uv run -- wiz-tts "Compressed audio" --data-dir ./saved_audio --bitrate 16k
```

## Features

- Converts text to speech using OpenAI's TTS API
- Real-time FFT (Fast Fourier Transform) visualization during playback
- Multiple voice options
- Custom speech style instructions
- Reads text from command line arguments or stdin
- Supports multiple TTS models
- Option to save generated audio as WebM files with metadata and configurable compression

## Requirements

- Python 3.12 or higher
- An OpenAI API key set in your environment variables as `OPENAI_API_KEY`

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `WIZ_TTS_DATA_DIR`: Default directory for saving audio files (optional, can be overridden with `--data-dir`)

## License

MIT
