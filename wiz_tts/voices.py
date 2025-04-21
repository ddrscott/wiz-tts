"""Voice configuration for TTS adapters."""

# Mapping of voices to their respective adapters
VOICE_ADAPTERS = {}

# Groq voices
GROQ_VOICES = [
    "Arista-PlayAI",
    "Atlas-PlayAI",
    "Basil-PlayAI",
    "Briggs-PlayAI",
    "Calum-PlayAI",
    "Celeste-PlayAI",
    "Cheyenne-PlayAI",
    "Chip-PlayAI",
    "Cillian-PlayAI",
    "Deedee-PlayAI",
    "Fritz-PlayAI",
    "Gail-PlayAI",
    "Indigo-PlayAI",
    "Mamaw-PlayAI",
    "Mason-PlayAI",
    "Mikail-PlayAI",
    "Mitch-PlayAI",
    "Quinn-PlayAI",
    "Thunder-PlayAI",
]

# OpenAI voices
OPENAI_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "onyx", "nova", "sage", "shimmer", "verse"
]

# Build the voice adapter mappings
for voice in GROQ_VOICES:
    VOICE_ADAPTERS[voice] = "groq_tts"

for voice in OPENAI_VOICES:
    VOICE_ADAPTERS[voice] = "openai_tts"

# Model overrides for specific adapters
MODEL_OVERRIDES = {
    "groq_tts": "playai-tts"
}
