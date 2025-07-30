import json
import os
from datetime import datetime

DATA_FILE = "disease_log.json"

def load_disease_logs():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_disease_report(image_url, disease_class, lat, lon):
    logs = load_disease_logs()
    new_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "image": image_url,
        "disease": disease_class,
        "latitude": lat,
        "longitude": lon
    }
    logs.append(new_entry)
    with open(DATA_FILE, "w") as f:
        json.dump(logs, f, indent=4)
