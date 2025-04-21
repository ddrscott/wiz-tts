from typing import AsyncIterator

# OpenAI returns 24kHz audio
SAMPLE_RATE = 24000

async def generate_speech(
    text: str,
    voice: str = "ash",
    instructions: str = "",
    model: str = "gpt-4o-mini-tts"
) -> AsyncIterator[bytes]:
    """
    Generate speech from text using OpenAI's TTS API.

    Args:
        text: The text to convert to speech
        voice: The voice to use
        instructions: Voice style instructions
        model: The TTS model to use

    Returns:
        An async iterator of audio chunks
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    async with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="wav",
    ) as response:
        async for chunk in response.iter_bytes(1024 * 8):  # Use smaller chunks for lower latency
            yield chunk
