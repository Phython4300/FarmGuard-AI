# transcribe.py
import whisper
import speech_recognition as sr
import os

def transcribe_with_whisper(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print("Whisper failed:", e)
        return None

def transcribe_with_google(audio_path):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print("Google Speech Recognition failed:", e)
        return None

def transcribe_audio(audio_path):
    print(f"Transcribing: {audio_path}")
    text = transcribe_with_whisper(audio_path)
    if text:
        return text.strip()

    # fallback
    text = transcribe_with_google(audio_path)
    if text:
        return text.strip()

    return "⚠️ Unable to transcribe audio."
