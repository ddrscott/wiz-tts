from typing import AsyncIterator
import asyncio

# Groq returns 48kHz audio
SAMPLE_RATE = 48000

async def generate_speech(
    text: str,
    voice: str = "Thunder-PlayAI",
    instructions: str = "",
    model: str = "playai-tts",
) -> AsyncIterator[bytes]:
    """
    Generate speech from text using Groq's API.

    Args:
        text: The text to convert to speech
        voice: The voice to use
        instructions: Voice style instructions (not used by Groq)
        model: The TTS model to use

    Returns:
        An async iterator of audio chunks
    """
    from groq import Groq

    client = Groq()

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        response_format="wav",
        input=text,
    )

    # Convert synchronous iterator to asynchronous iterator
    for chunk in response.iter_bytes(1024 * 8):
        yield chunk
        # Small delay to allow other tasks to run
        await asyncio.sleep(0)
