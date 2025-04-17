# Wiz TTS

A simple command-line tool for text-to-speech using OpenAI's API, featuring real-time FFT visualization.

## Installation

```bash
pip install wiz-tts
```

## Usage

After installation, you can run the tool with:

```bash
wiz-tts "Your text to convert to speech"
```

### Options

```
usage: wiz-tts [-h] [--voice {alloy,echo,fable,onyx,nova,shimmer,coral}] [--instructions INSTRUCTIONS] [text]

Convert text to speech with visualization

positional arguments:
  text                  Text to convert to speech (default: "Today is a wonderful day to build something people love!")

options:
  -h, --help            show this help message and exit
  --voice {alloy,echo,fable,onyx,nova,shimmer,coral}, -v {alloy,echo,fable,onyx,nova,shimmer,coral}
                        Voice to use for speech (default: coral)
  --instructions INSTRUCTIONS, -i INSTRUCTIONS
                        Instructions for the speech style
```

### Examples

Basic usage:
```bash
wiz-tts "Hello, world!"
```

Using a different voice:
```bash
wiz-tts --voice nova "Welcome to the future of text to speech!"
```

Adding speech instructions:
```bash
wiz-tts --voice shimmer --instructions "Speak slowly and clearly" "This is important information."
```

## Features

- Converts text to speech using OpenAI's TTS API
- Real-time FFT (Fast Fourier Transform) visualization during playback
- Multiple voice options
- Custom speech style instructions

## Requirements

- Python 3.12 or higher
- An OpenAI API key set in your environment variables as `OPENAI_API_KEY`

## License

MIT