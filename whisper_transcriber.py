# whisper_transcriber.py

import openai
import os
import tempfile

def transcribe_audio(audio_file, api_key):
    """
    Transcribes an audio file using OpenAI Whisper API.
    :param audio_file: The uploaded file (streamlit object)
    :param api_key: Your OpenAI API key
    :return: Transcribed text or error message
    """
    try:
        openai.api_key = api_key

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        # Open it again for transcription
        with open(tmp_file_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)

        os.remove(tmp_file_path)
        return transcript["text"]

    except Exception as e:
        return f"Transcription failed: {str(e)}"
