import pyttsx3

engine = pyttsx3.init()

# Get all available voices
voices = engine.getProperty('voices')

# Print available voices
for index, voice in enumerate(voices):
    print(f"Voice {index}: {voice.name}")
    print(f"  - ID: {voice.id}")
    print(f"  - Languages: {voice.languages}")
    print()

# Set a specific voice (by index)
engine.setProperty('voice', voices[1].id)  # Usually female voice

engine.say("Now I'm speaking with a different voice!")
engine.runAndWait()