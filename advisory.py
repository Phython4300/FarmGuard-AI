from gtts import gTTS
import os
import tempfile

# Local advisory messages (expand this as needed)
advisory_messages = {
    "Healthy": "Shukar ka lafiya lau ce. Ci gaba da kula da ita.",
    "CMD": "Shukar ka na dauke da cutar Cassava Mosaic Disease. A cire ganyen da suka kamu kuma a shuka sabbi masu lafiya.",
    "CBSD": "Shukar ka na dauke da Cassava Brown Streak Disease. A tuntubi masanin noma domin shawara.",
    "Brown Spots": "An gano tabo a ganyen shukar ka. A iya amfani da maganin fungi."
}

def get_advisory(disease_class):
    return advisory_messages.get(disease_class, "Ba a gane cutar ba. A tuntubi masani.")

def generate_audio_advisory(message, lang='ha'):
    tts = gTTS(text=message, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name
