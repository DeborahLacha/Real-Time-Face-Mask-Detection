# app.py
import os
import time
import math
import tempfile
import threading
import csv
from collections import OrderedDict, deque, Counter
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration, WebRtcMode
import av


import os
import requests
import streamlit as st

# ---------- Robust model-path resolution ----------
def get_model_url_from_secrets_or_env():
    # Try Streamlit secrets safely (may raise StreamlitSecretNotFoundError if no secrets file)
    try:
        # try dictionary-style access (avoids .get calling internal parse)
        if "MODEL_URL" in st.secrets:
            return st.secrets["MODEL_URL"]
    except Exception:
        # no secrets configured or parsing error; ignore
        pass
    # fallback: environment variable
    env_url = os.environ.get("MODEL_URL")
    if env_url:
        return env_url
    # no remote model configured
    return None

@st.cache_resource
def download_model_from_url(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return dest_path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest_path

# determine model path
MODEL_URL = get_model_url_from_secrets_or_env()
if MODEL_URL:
    model_local_path = os.path.join("models", os.path.basename(MODEL_URL))
    try:
        model_local_path = download_model_from_url(MODEL_URL, model_local_path)
        st.sidebar.success("Model downloaded")
    except Exception as e:
        st.sidebar.error(f"Failed to download model from MODEL_URL: {e}")
        model_local_path = None
else:
    # fallback to local path (no crash)
    model_local_path = "mask_detector.model" if os.path.exists("mask_detector.model") else None

# model_local_path is either the downloaded path, a local file path, or None


# Optional: put your ST Cloud key here if required (most cases not needed)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(page_title="Mask Compliance Monitor", layout="wide")
st.title("Mask Compliance Monitor — Streamlit")

# -------------------------
# Sidebar: model selection
# -------------------------
st.sidebar.header("Model & Settings")

model_file = st.sidebar.file_uploader("Upload model (.h5/.model or .tflite). If blank, use local path 'mask_detector.model'", type=["h5","model","tflite"])
use_tflite = st.sidebar.checkbox("Model is TFLite", value=False)
model_path_input = st.sidebar.text_input("Or local model path", value="mask_detector.model")

width = st.sidebar.slider("Processing width (px)", 240, 800, 400, step=40)
process_every_n = st.sidebar.slider("Process every Nth frame (speed)", 1, 8, 2)
history_len = st.sidebar.slider("Label history length (smoothing)", 1, 9, 5)
nms_thresh = st.sidebar.slider("NMS IoU threshold", 0.1, 0.6, 0.3)

alarm_file = st.sidebar.file_uploader("Optional: Upload alarm sound (wav/mp3)", type=["wav","mp3"])
debounce_seconds = st.sidebar.number_input("Alert debounce (s)", min_value=0, max_value=60, value=5)

log_path = st.sidebar.text_input("CSV log path", value="streamlit_detection_log.csv")
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: if using a large Keras model, prefer converting it to a TFLite int8 for speed.")

# Save uploaded model & alarm to temp files (if provided)
tmpdir = tempfile.gettempdir()
uploaded_model_path = None
if model_file is not None:
    uploaded_model_path = os.path.join(tmpdir, model_file.name)
    with open(uploaded_model_path, "wb") as f:
        f.write(model_file.getbuffer())

uploaded_alarm_path = None
if alarm_file is not None:
    uploaded_alarm_path = os.path.join(tmpdir, alarm_file.name)
    with open(uploaded_alarm_path, "wb") as f:
        f.write(alarm_file.getbuffer())

# If user didn't upload, fallback to text path
if uploaded_model_path is None:
    if os.path.exists(model_path_input):
        model_path = model_path_input
    else:
        model_path = None
else:
    model_path = uploaded_model_path

# -------------------------
# Helper functions
# -------------------------
def append_log(csv_path, row):
    header = ['timestamp', 'id', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2']
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        st.sidebar.error(f"Failed to write log: {e}")

# Small centroid tracker (same idea as earlier)
class SimpleCentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=80):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.labels = {}
        self.last_conf = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        oid = self.nextObjectID; self.nextObjectID += 1
        self.objects[oid] = centroid; self.disappeared[oid] = 0
        self.labels[oid] = None; self.last_conf[oid] = 0.0
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]; del self.disappeared[oid]
            if oid in self.labels: del self.labels[oid]
            if oid in self.last_conf: del self.last_conf[oid]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects
        objectIDs = list(self.objects.keys()); objectCentroids = list(self.objects.values())
        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")
        for i in range(len(objectCentroids)):
            for j in range(len(input_centroids)):
                D[i, j] = math.hypot(objectCentroids[i][0] - input_centroids[j][0],
                                     objectCentroids[i][1] - input_centroids[j][1])
        rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[rows]
        usedRows, usedCols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols: continue
            if D[row, col] > self.max_distance: continue
            oid = objectIDs[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            usedRows.add(row); usedCols.add(col)
        for j in range(len(input_centroids)):
            if j not in usedCols:
                self.register(input_centroids[j])
        for i in range(len(objectCentroids)):
            if i not in usedRows:
                oid = objectIDs[i]; self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
        return self.objects

# NMS helper using OpenCV's NMSBoxes (rects must be x,y,w,h)
def apply_nms(locs, scores, nms_thresh=0.3, score_thresh=0.0):
    if len(locs) == 0:
        return [], []
    rects = []
    for (x1,y1,x2,y2) in locs:
        rects.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
    idxs = cv2.dnn.NMSBoxes(rects, scores, score_thresh, nms_thresh)
    kept, filtered = [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,w,h = rects[i]; filtered.append((x,y,x+w,y+h)); kept.append(i)
    return filtered, kept

# -------------------------
# Video transformer (streamlit-webrtc)
# -------------------------
class MaskTransformer(VideoTransformerBase):
    def __init__(self):
        self.ready = False
        self.use_tflite = use_tflite
        self.model_path = model_path
        self.width = width
        self.process_every = process_every_n
        self.frame_no = 0
        self.tracker = SimpleCentroidTracker(max_disappeared=40, max_distance=80)
        self.history_len = history_len
        self.label_history = {}  # oid -> deque(...)
        self.seen_ids = set()
        self.last_alert_time = {}
        # Haar cascade for portability
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.nms_thresh = nms_thresh

        # load model if available
        if self.model_path:
            try:
                if self.use_tflite and self.model_path.lower().endswith(".tflite"):
                    try:
                        import tflite_runtime.interpreter as tflite
                        Interpreter = tflite.Interpreter
                    except Exception:
                        # fallback to TF interpreter if tflite-runtime missing
                        from tensorflow.lite import Interpreter
                        Interpreter = Interpreter
                    self.interpreter = Interpreter(model_path=self.model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    st.sidebar.success("TFLite model loaded")
                else:
                    # load Keras/TensorFlow model (may be heavy)
                    import tensorflow as tf
                    self.keras_model = tf.keras.models.load_model(self.model_path)
                    st.sidebar.success("Keras model loaded")
                self.ready = True
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")
                self.ready = False
        else:
            st.sidebar.warning("No model found — upload or set local model path.")

    def preprocess(self, crop):
        img = cv2.resize(crop, (224,224)).astype("float32")
        img = (img / 127.5) - 1.0
        return img

    def predict_batch(self, crops):
        # crops: list of BGR arrays
        if len(crops) == 0:
            return []
        batch = np.stack([self.preprocess(c) for c in crops], axis=0)
        if getattr(self, "interpreter", None) is not None:
            inp = batch
            input_dtype = self.input_details[0]['dtype']
            q = self.input_details[0].get('quantization', (0.0, 0))
            scale, zp = q
            if input_dtype in [np.int8, np.uint8] and scale != 0:
                inp_q = (inp / scale + zp).astype(input_dtype)
            else:
                inp_q = inp.astype(input_dtype)
            self.interpreter.set_tensor(self.input_details[0]['index'], inp_q)
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_details[0]['index'])
            if out.dtype in [np.int8, np.uint8]:
                o_scale, o_zero = self.output_details[0].get('quantization', (0.0,0))
                if o_scale != 0:
                    out = (out.astype(np.float32) - o_zero) * o_scale
            return out
        else:
            preds = self.keras_model.predict(batch)
            return preds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_no += 1
        orig = img.copy()

        # optionally skip frames to save CPU
        if (self.frame_no % self.process_every) != 0:
            # still show overlayed HUD using last known tracker state
            self.draw_hud(orig)
            return av.VideoFrame.from_ndarray(orig, format="bgr24")

        # detect faces with Haar cascade (scale down for speed)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))

        locs = []
        for (x,y,w,h) in faces:
            locs.append((x, y, x+w, y+h))

        # If no faces detected, update tracker with empty list
        if len(locs) == 0:
            self.tracker.update([])
            self.draw_hud(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Prepare crops for model
        crops = []
        for (sx, sy, ex, ey) in locs:
            crop = img[sy:ey, sx:ex]
            if crop.size == 0:
                crops.append(None)
            else:
                crops.append(crop)

        # get predictions (batch)
        preds = self.predict_batch([c for c in crops if c is not None])

        # map predictions back to locs
        preds_full = [None] * len(locs)
        bi = 0
        for i in range(len(crops)):
            if crops[i] is None:
                preds_full[i] = None
            else:
                if bi < len(preds):
                    preds_full[i] = preds[bi]; bi += 1

        # compute scores for NMS: use max(prob) to prefer higher-conf detections
        scores = []
        for p in preds_full:
            if p is None:
                scores.append(1.0)
            else:
                scores.append(float(np.max(p)))

        filtered_boxes, kept_idx = apply_nms(locs, scores, nms_thresh=self.nms_thresh)

        # calc centroids & update tracker
        centroids = []
        for (sx, sy, ex, ey) in filtered_boxes:
            centroids.append((int((sx+ex)/2), int((sy+ey)/2)))
        objects = self.tracker.update(centroids)

        # mapping for quick lookup
        centroid_to_box = {centroids[i]: filtered_boxes[i] for i in range(len(centroids))}

        # process each present tracked object
        for oid, cent in list(objects.items()):
            if cent not in centroid_to_box:
                continue
            (sx, sy, ex, ey) = centroid_to_box[cent]
            crop = img[sy:ey, sx:ex]
            if crop.size == 0:
                continue

            # find pred index by matching coordinates in locs (best-effort)
            pred_idx = None
            for i, b in enumerate(locs):
                if b == (sx, sy, ex, ey):
                    pred_idx = i
                    break

            label = "Unknown"; conf = 0.0
            if pred_idx is not None and preds_full[pred_idx] is not None:
                out = preds_full[pred_idx]
                if out.shape[0] == 2:
                    # binary: [mask, withoutMask] (older formats)
                    if float(out[0]) > float(out[1]):
                        label, conf = "Mask", float(out[0])
                    else:
                        label, conf = "No Mask", float(out[1])
                elif out.shape[0] == 3:
                    cid = int(np.argmax(out)); conf = float(out[cid])
                    label = ["Mask", "No Mask", "Incorrect Mask"][cid]
                else:
                    cid = int(np.argmax(out)); conf = float(out[cid]); label = f"Class_{cid}"
            else:
                # fallback single predict (should be rare)
                try:
                    p = self.predict_batch([crop])[0]
                    if p.shape[0] == 2:
                        if float(p[0]) > float(p[1]):
                            label, conf = "Mask", float(p[0])
                        else:
                            label, conf = "No Mask", float(p[1])
                    elif p.shape[0] == 3:
                        cid = int(np.argmax(p)); conf = float(p[cid]); label = ["Mask","No Mask","Incorrect Mask"][cid]
                except Exception:
                    label, conf = "Err", 0.0

            # smoothing history
            if oid not in self.label_history:
                self.label_history[oid] = deque(maxlen=self.history_len)
            self.label_history[oid].append(label)
            smoothed = Counter(self.label_history[oid]).most_common(1)[0][0]

            self.tracker.labels[oid] = smoothed
            self.tracker.last_conf[oid] = conf
            self.seen_ids.add(oid)

            # append CSV log (UTC)
            append_log(log_path, [datetime.utcnow().isoformat(), oid, smoothed, f"{conf:.4f}", sx, sy, ex, ey])

            # alert (debounced)
            if smoothed != "Mask":
                now = time.time()
                if oid not in self.last_alert_time or (now - self.last_alert_time[oid]) > debounce_seconds:
                    self.last_alert_time[oid] = now
                    # play or note an alert (audio only works on local machine)
                    if uploaded_alarm_path:
                        threading.Thread(target=play_sound, args=(uploaded_alarm_path,), daemon=True).start()

            # draw on frame
            color = (0,255,0) if smoothed == "Mask" else (0,165,255) if "Incorrect" in smoothed else (0,0,255)
            text = f"ID {oid}: {smoothed} {conf:.2f}"
            cv2.putText(img, text, (sx, max(15, sy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(img, (sx, sy), (ex, ey), color, 2)

        # draw HUD info computed from present objects (centroid_in_box)
        present_oids = [oid for oid, cent in self.tracker.objects.items() if cent in centroid_to_box]
        mask_ct = no_ct = inc_ct = 0
        for oid in present_oids:
            lab = self.tracker.labels.get(oid, "").lower()
            if "incorrect" in lab:
                inc_ct += 1
            elif "mask" in lab and "incorrect" not in lab:
                mask_ct += 1
            elif "no" in lab:
                no_ct += 1
        current_present = len(present_oids)
        compliance = (mask_ct / current_present * 100.0) if current_present > 0 else 0.0
        cv2.putText(img, f"Present: {current_present}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Mask:{mask_ct} No:{no_ct} Inc:{inc_ct}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Compliance:{compliance:.1f}% Seen:{len(self.seen_ids)}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# small helper to play sound on local machine
def play_sound(path):
    try:
        from playsound import playsound
        playsound(path)
    except Exception:
        try:
            import simpleaudio as sa
            wave_obj = sa.WaveObject.from_wave_file(path)
            wave_obj.play()
        except Exception:
            pass

# -------------------------
# UI layout and start
# -------------------------
col1, col2 = st.columns([2,1])
with col1:
    webrtc_ctx = webrtc_streamer(
        key="mask-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=MaskTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

with col2:
    st.header("Live Stats")
    stats_placeholder = st.empty()
    st.markdown("**Controls**")
    st.write(f"- Model path: `{model_path}`")
    st.write(f"- TFLite mode: {use_tflite}")
    st.write(f"- Width: {width}px, process every {process_every_n} frames")
    st.write(f"- NMS IoU: {nms_thresh}, history_len: {history_len}")

if webrtc_ctx.state.playing:
    st.info("Camera is running. Give the browser permission to use camera.")
    # Poll transformer to update stats in sidebar
    while webrtc_ctx.state.playing:
        transformer = webrtc_ctx.video_transformer
        if transformer and hasattr(transformer, "tracker"):
            # present counts
            active = [oid for oid, c in transformer.tracker.objects.items() if c]
            present_oids = [oid for oid in active if oid in transformer.label_history]
            mask_ct = no_ct = inc_ct = 0
            for oid in present_oids:
                lab = transformer.tracker.labels.get(oid, "").lower()
                if "incorrect" in lab: inc_ct += 1
                elif "mask" in lab and "incorrect" not in lab: mask_ct += 1
                elif "no" in lab: no_ct += 1
            total_present = len(present_oids)
            compliance = (mask_ct / total_present * 100.0) if total_present > 0 else 0.0
            stats_placeholder.markdown(f"""
            **Present:** {total_present}  
            **Mask:** {mask_ct} · **No Mask:** {no_ct} · **Incorrect:** {inc_ct}  
            **Compliance:** {compliance:.1f}% · **Unique seen:** {len(transformer.seen_ids)}
            """)
        time.sleep(0.8)
else:
    st.warning("Start the camera by allowing the browser to use your webcam and press 'Start' if prompted.")
