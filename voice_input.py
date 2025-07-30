# voice_input.py
import streamlit as st
import os

def record_or_upload_audio():
    st.subheader("🎙️ Upload Your Voice Message")
    audio_file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

    if audio_file is not None:
        # Save uploaded file to temp
        audio_path = os.path.join("temp", audio_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        return audio_path

    return None
