# app.py
import os, json, uuid, time
import numpy as np
import librosa
import torch
import requests

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

import tensorflow as tf
import tensorflow_hub as hub
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# =========================================================
# CONFIG 
# =========================================================
OUT_DIR = r"C:\Users\hajid\Documents\project"
WEIGHTS_PATH = os.path.join(OUT_DIR, "best_weights.weights.h5")
LABELS_PATH = os.path.join(OUT_DIR, "labels.json")

# fichiers emotion/sentiment
EMO_MODEL_PATH = "wav2vec2_crema_best.pth"
EMO_LABELS_PATH = "crema_labels.json"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"wav", "mp3", "ogg", "flac"}

# scream preprocessing
SR = 16000
DURATION = 3.0
MAX_SAMPLES = int(SR * DURATION)

# emotion preprocessing
EMO_SR = 16000
EMO_MAX_DURATION = 2.5

# =========================
# Node-RED webhook config
# =========================
NODERED_URL = "http://10.104.31.165:1880/ai/detect"

# active/désactive l'envoi
SEND_TO_NODERED = True

# sécurité (optionnel)
USE_NODERED_API_KEY = False
NODERED_API_KEY = "mysecret123"  # change si tu actives

# robustesse réseau
NODERED_RETRIES = 0
NODERED_TIMEOUT_SEC = 10  # 0/1/2...
NODERED_RETRY_SLEEP = 0.5  # secondes

# =========================================================
# CHECK FILES
# =========================================================
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"labels.json introuvable: {LABELS_PATH}")
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"weights introuvable: {WEIGHTS_PATH}")
if not os.path.exists(EMO_LABELS_PATH):
    raise FileNotFoundError(f"emo labels introuvable: {EMO_LABELS_PATH}")
if not os.path.exists(EMO_MODEL_PATH):
    raise FileNotFoundError(f"emo model .pth introuvable: {EMO_MODEL_PATH}")

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)  

with open(EMO_LABELS_PATH, "r") as f:
    EMO_LABELS = json.load(f)  

# =========================================================
# LOAD SCREAM MODEL (YAMNet + head)
# =========================================================
yamnet = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)

class YamnetEmbed(tf.keras.layers.Layer):
    def __init__(self, yamnet_layer, **kwargs):
        super().__init__(**kwargs)
        self.yamnet_layer = yamnet_layer

    def call(self, wave_batch):
        def one(w):
            _, embeddings, _ = self.yamnet_layer(w)
            return tf.reduce_mean(embeddings, axis=0)
        return tf.map_fn(one, wave_batch, fn_output_signature=tf.float32)

embed_layer = YamnetEmbed(yamnet)

inp = tf.keras.Input(shape=(MAX_SAMPLES,), dtype=tf.float32, name="waveform")
x = embed_layer(inp)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(2, activation="softmax")(x)

scream_model = tf.keras.Model(inp, out)
scream_model.load_weights(WEIGHTS_PATH)
scream_model.trainable = False

print("✅ Scream model loaded")

# =========================================================
# LOAD EMOTION MODEL (Wav2Vec2)
# =========================================================
emo_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

emo_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(EMO_LABELS),
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(EMO_MODEL_PATH, map_location="cpu")
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    emo_model.load_state_dict(checkpoint["model_state_dict"])
else:
    emo_model.load_state_dict(checkpoint)

emo_model.eval()
print("✅ Emotion model loaded")

# =========================================================
# HELPERS
# =========================================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_upload(file_storage):
    filename = secure_filename(file_storage.filename)
    if not filename or not allowed_file(filename):
        return None, "Invalid file type (wav/mp3/ogg/flac only)"
    ext = filename.rsplit(".", 1)[1].lower()
    new_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, new_name)
    file_storage.save(path)
    return path, None

def _label_from_index(labels_obj, idx: int):
    if isinstance(labels_obj, dict):
        if str(idx) in labels_obj:
            return labels_obj[str(idx)]
        if idx in labels_obj:
            return labels_obj[idx]
        return labels_obj.get(str(idx), labels_obj.get(idx, str(idx)))
    else:
        return labels_obj[idx]

def send_to_nodered(payload: dict):
    if not SEND_TO_NODERED:
        return {"sent": False, "reason": "SEND_TO_NODERED disabled"}

    headers = {}
    if USE_NODERED_API_KEY:
        headers["X-API-KEY"] = NODERED_API_KEY

    last_err = None
    for attempt in range(NODERED_RETRIES + 1):
        try:
            r = requests.post(
                NODERED_URL, json=payload, headers=headers, timeout=NODERED_TIMEOUT_SEC
            )
            ok = (200 <= r.status_code < 300)
            print(f"✅ Node-RED attempt {attempt+1}: status={r.status_code}, ok={ok}")
            if not ok:
                print("⚠️ Node-RED response:", (r.text or "")[:200])
            return {
                "sent": ok,
                "status_code": r.status_code,
                "response": (r.text or "")[:200]
            }
        except Exception as e:
            last_err = str(e)
            print(f"⚠️ Node-RED attempt {attempt+1} failed:", last_err)
            if attempt < NODERED_RETRIES:
                time.sleep(NODERED_RETRY_SLEEP)

    return {"sent": False, "error": last_err}

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def preprocess_scream(audio_path: str) -> np.ndarray:
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    if len(wav) < MAX_SAMPLES:
        wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
    else:
        wav = wav[:MAX_SAMPLES]
    return wav.astype(np.float32)

def predict_scream(audio_path: str):
    wav = preprocess_scream(audio_path)
    probs = scream_model(np.expand_dims(wav, axis=0)).numpy()[0]
    idx = int(np.argmax(probs))
    label = _label_from_index(LABELS, idx)
    return label, float(probs[idx]), probs.tolist()

def predict_emotion(audio_path: str):
    audio, _ = librosa.load(audio_path, sr=EMO_SR, mono=True)
    max_len = int(EMO_SR * EMO_MAX_DURATION)

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = torch.nn.functional.pad(
            torch.tensor(audio), (0, max_len - len(audio))
        ).numpy()

    inputs = emo_processor(
        audio, sampling_rate=EMO_SR, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = emo_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        emo_label = _label_from_index(EMO_LABELS, idx)

    return emo_label, float(probs[idx])

# =========================================================
# FLASK APP + UI
# =========================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HOME_HTML = """ ... """
RESULT_HTML = """ ... """

@app.route("/test_nodered", methods=["GET"])
def test_nodered():
    payload = {
        "source": "flask_test",
        "scream_prediction": "scream",
        "scream_confidence": 0.91,
        "emotion": "angry",
        "emotion_confidence": 0.72,
    }
    nr = send_to_nodered(payload)
    return jsonify({"to": NODERED_URL, "payload": payload, "node_red": nr})

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return "No file uploaded", 400

    f = request.files["audio"]
    path, err = save_upload(f)
    if err:
        return err, 400

    try:
        scream_label, scream_conf, _ = predict_scream(path)
        emo_label, emo_conf = predict_emotion(path)
    except Exception as e:
        try: os.remove(path)
        except: pass
        return f"Prediction error: {e}", 500

    try: os.remove(path)
    except: pass

    payload = {
        "source": "web",
        "scream_prediction": scream_label,
        "scream_confidence": round(float(scream_conf), 4),
        "emotion": emo_label,
        "emotion_confidence": round(float(emo_conf), 4),
    }
    nr = send_to_nodered(payload)

    return render_template_string(
        RESULT_HTML,
        scream_label=scream_label,
        scream_conf=round(float(scream_conf), 4),
        emo_label=emo_label,
        emo_conf=round(float(emo_conf), 4),
        nr_sent=nr.get("sent", False),
        nr_status=nr.get("status_code", None),
    )

@app.route("/predict", methods=["POST"])
def predict_api():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    path, err = save_upload(f)
    if err:
        return jsonify({"error": err}), 400

    try:
        scream_label, scream_conf, scream_probs = predict_scream(path)
        emo_label, emo_conf = predict_emotion(path)
    except Exception as e:
        try: os.remove(path)
        except: pass
        return jsonify({"error": str(e)}), 500

    try: os.remove(path)
    except: pass

    payload = {
        "source": "api",
        "scream_prediction": scream_label,
        "scream_confidence": round(float(scream_conf), 4),
        "emotion": emo_label,
        "emotion_confidence": round(float(emo_conf), 4),
    }
    nr = send_to_nodered(payload)

    return jsonify({
        "scream_prediction": scream_label,
        "scream_confidence": round(float(scream_conf), 4),
        "scream_probs": [round(float(p), 6) for p in scream_probs],
        "emotion": emo_label,
        "emotion_confidence": round(float(emo_conf), 4),
        "node_red": nr
    })

@app.route("/esp32", methods=["POST"])
def esp32_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    path, err = save_upload(f)
    if err:
        return jsonify({"error": err}), 400

    try:
        scream_label, scream_conf, _ = predict_scream(path)
        emo_label, emo_conf = predict_emotion(path)
    except Exception as e:
        try: os.remove(path)
        except: pass
        return jsonify({"error": str(e)}), 500

    try: os.remove(path)
    except: pass

    payload = {
        "source": "esp32",
        "scream_prediction": scream_label,
        "scream_confidence": round(float(scream_conf), 4),
        "emotion": emo_label,
        "emotion_confidence": round(float(emo_conf), 4),
    }
    nr = send_to_nodered(payload)

    payload_out = dict(payload)
    payload_out["node_red"] = nr
    return jsonify(payload_out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)