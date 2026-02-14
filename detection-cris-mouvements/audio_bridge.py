import time
import numpy as np
import sounddevice as sd
import soundfile as sf

from app import predict_scream, predict_emotion, send_to_nodered

# ---------------------------
# CONFIG
# ---------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
DEVICE_NAME_CONTAINS = "WO Mic"  

def find_input_device(name_contains: str):
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d["max_input_channels"] > 0 and name_contains.lower() in d["name"].lower():
            return idx, d["name"]
    return None, None

def list_input_devices():
    devices = sd.query_devices()
    print("🎙️ Périphériques d'entrée disponibles:")
    for idx, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f" [{idx}] {d['name']} (in={d['max_input_channels']})")

def record_wav(filename: str, device_idx: int):
    print(f"🔴 Enregistrement {RECORD_SECONDS}s... ({filename})")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device_idx
    )
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE, subtype="PCM_16")
    print("💾 Sauvegardé:", filename)

def main():
    list_input_devices()
    dev_idx, dev_name = find_input_device(DEVICE_NAME_CONTAINS)

    if dev_idx is None:
        print(f"\n❌ Micro contenant '{DEVICE_NAME_CONTAINS}' introuvable.")
        print("➡️ Change DEVICE_NAME_CONTAINS (ex: 'WO Mic' ou 'Droidcam Audio') ou choisis l'index manuellement.")
        return

    print(f"\n✅ Micro détecté: [{dev_idx}] {dev_name}")

    while True:
        filename = f"audio_{int(time.time())}.wav"

        # 1) Enregistrer
        record_wav(filename, dev_idx)

        # 2) IA
        print("🎯 Analyse IA en cours...")
        scream_label, scream_conf, _ = predict_scream(filename)
        emo_label, emo_conf = predict_emotion(filename)

        print(f"🚨 Scream: {scream_label} ({scream_conf:.3f})")
        print(f"😊 Emotion: {emo_label} ({emo_conf:.3f})")

        # 3) Node-RED
        payload = {
            "source": "phone_wifi_mic",
            "file": filename,
            "scream_prediction": scream_label,
            "scream_confidence": round(float(scream_conf), 4),
            "emotion": emo_label,
            "emotion_confidence": round(float(emo_conf), 4),
        }
        nr = send_to_nodered(payload)
        print("📡 Envoyé à Node-RED:", nr)

        time.sleep(0.2)

if __name__ == "__main__":
    main()