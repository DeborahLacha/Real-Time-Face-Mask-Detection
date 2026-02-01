#!/usr/bin/env python3
# USAGE
# python detect_mask_video.py --model code/mask_detector.model --face face_detector --alarm alarm.mp3

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import math
import csv
import threading
from collections import OrderedDict
from datetime import datetime

# optional sound playback libs (best-effort)
try:
    from playsound import playsound
    def _play_sound(path):
        playsound(path)
except Exception:
    try:
        import simpleaudio as sa
        def _play_sound(path):
            wave_obj = sa.WaveObject.from_wave_file(path)
            wave_obj.play()
    except Exception:
        _play_sound = None

# Optional TFLite interpreter import helper
def import_tflite_interpreter():
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter
            return Interpreter
        except Exception:
            return None

# ---------- Simple centroid tracker ----------
class SimpleCentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()       # objectID -> centroid (x,y)
        self.disappeared = OrderedDict()   # objectID -> disappeared frames
        self.labels = {}                   # objectID -> last label
        self.last_conf = {}                # objectID -> last confidence
        self.last_seen = {}                # objectID -> last seen frame index
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, frame_no):
        oid = self.nextObjectID
        self.nextObjectID += 1
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.labels[oid] = None
        self.last_conf[oid] = 0.0
        self.last_seen[oid] = frame_no
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
            del self.disappeared[oid]
            del self.labels[oid]
            del self.last_conf[oid]
            del self.last_seen[oid]

    def update(self, input_centroids, frame_no):
        # if no detections: mark disappeared and maybe deregister
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # if we have no objects currently, register all input centroids
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c, frame_no)
            return self.objects

        # otherwise compute distance matrix between existing objects and new inputs
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")

        for i in range(len(objectCentroids)):
            for j in range(len(input_centroids)):
                D[i, j] = math.hypot(objectCentroids[i][0] - input_centroids[j][0],
                                     objectCentroids[i][1] - input_centroids[j][1])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue

            oid = objectIDs[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            self.last_seen[oid] = frame_no

            usedRows.add(row)
            usedCols.add(col)

        # register any unused input centroids as new objects
        for j in range(len(input_centroids)):
            if j not in usedCols:
                self.register(input_centroids[j], frame_no)

        # mark disappeared objects
        for i in range(len(objectCentroids)):
            if i not in usedRows:
                oid = objectIDs[i]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        return self.objects

# ---------- detection + prediction function (keeps your original approach) ----------
def detect_and_predict_mask(frame, faceNet, maskNet, args, use_tflite=False, tflite_interpreter=None):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # ensure region is valid
            if endX <= startX or endY <= startY:
                continue

            face = frame[startY:endY, startX:endX]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_arr = img_to_array(face_rgb)
            face_arr = preprocess_input(face_arr)
            face_arr = np.expand_dims(face_arr, axis=0)

            faces.append(face_arr)
            locs.append((startX, startY, endX, endY))

    preds = []
    if len(faces) > 0:
        # stack faces into a single batch
        face_batch = np.vstack(faces)  # shape (N,224,224,3)

        # TFLite path
        if use_tflite and tflite_interpreter is not None:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            inp_dtype = input_details[0]['dtype']
            # handle quantization input if applicable
            qparams = input_details[0].get('quantization', (0.0, 0))
            scale, zero_point = qparams
            if inp_dtype in [np.int8, np.uint8] and scale != 0:
                in_data = (face_batch / scale + zero_point).astype(inp_dtype)
            else:
                in_data = face_batch.astype(inp_dtype)
            tflite_interpreter.set_tensor(input_details[0]['index'], in_data)
            tflite_interpreter.invoke()
            out = tflite_interpreter.get_tensor(output_details[0]['index'])
            # dequantize outputs if necessary
            out_dtype = output_details[0]['dtype']
            if out_dtype in [np.int8, np.uint8]:
                o_scale, o_zero = output_details[0].get('quantization', (0.0, 0))
                if o_scale != 0:
                    out = (out.astype(np.float32) - o_zero) * o_scale
            # ensure shape (N, classes)
            preds = out
        else:
            # Keras predict: returns shape (N, classes)
            preds = maskNet.predict(face_batch, batch_size=32)

    return (locs, preds)

# ---------- small helpers ----------
def append_log(csv_path, row):
    header = ['timestamp_utc', 'id', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2']
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def play_alert(path):
    if _play_sound is None:
        return
    threading.Thread(target=lambda: _play_sound(path), daemon=True).start()

# ----------------- main -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="code/mask_detector.model",
        help="path to trained face mask detector model (.h5 or .tflite)")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("--alarm", type=str, default=None, help="path to alarm sound file (mp3/wav)")
    ap.add_argument("--log", type=str, default="detection_log.csv", help="path to CSV event log")
    ap.add_argument("--debounce", type=int, default=5, help="seconds to debounce alerts per person")
    ap.add_argument("--use_tflite", action='store_true', help="treat model as TFLite")
    ap.add_argument("--width", type=int, default=400, help="resize width for processing")
    ap.add_argument("--max_disappeared", type=int, default=50, help="frames before deregistering")
    args = vars(ap.parse_args())

    # load face detector (OpenCV DNN)
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        raise FileNotFoundError("Face detector files not found in %s" % args["face"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load mask model (.h5 or tflite)
    use_tflite = args['use_tflite']
    tflite_interpreter = None
    maskNet = None
    if use_tflite:
        Interpreter = import_tflite_interpreter()
        if Interpreter is None:
            raise RuntimeError("TFLite interpreter not available. Install tflite-runtime or TensorFlow.")
        print("[INFO] loading TFLite model...")
        tflite_interpreter = Interpreter(model_path=args['model'])
        tflite_interpreter.allocate_tensors()
    else:
        print("[INFO] loading face mask detector model...")
        maskNet = load_model(args["model"])

    # start video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    tracker = SimpleCentroidTracker(max_disappeared=args['max_disappeared'], max_distance=50)
    last_alert_time = {}   # objectID -> timestamp
    seen_ids = set()       # unique ids seen
    counts = {'mask':0, 'no_mask':0, 'incorrect_mask':0, '_seen_ids': set()}

    frame_no = 0
    try:
        while True:
            frame = vs.read()
            if frame is None:
                break
            frame = imutils.resize(frame, width=args['width'])
            frame_no += 1

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, args,
                                                   use_tflite=use_tflite, tflite_interpreter=tflite_interpreter)

            # build centroids mapping used by tracker update
            centroids = []
            for (startX, startY, endX, endY) in locs:
                cx = int((startX + endX) / 2.0)
                cy = int((startY + endY) / 2.0)
                centroids.append((cx, cy))

            objects = tracker.update(centroids, frame_no)

            # map centroid -> box index for quick lookup
            centroid_to_box = {}
            for idx, c in enumerate(centroids):
                centroid_to_box[c] = locs[idx]

            # process tracked objects
            for oid, centroid in list(objects.items()):
                # try to find a detection that matches this centroid
                box = centroid_to_box.get(centroid, None)
                if box is None:
                    # draw a small dot for tracked object without current detection
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, (200,200,0), -1)
                    continue

                (startX, startY, endX, endY) = box
                # crop with small padding
                pad = 5
                sx = max(0, startX - pad); sy = max(0, startY - pad)
                ex = min(frame.shape[1] - 1, endX + pad); ey = min(frame.shape[0] - 1, endY + pad)
                face_crop = frame[sy:ey, sx:ex]
                if face_crop.size == 0:
                    continue

                # find index of this box in locs to get pred
                try:
                    idx = locs.index((startX, startY, endX, endY))
                except ValueError:
                    idx = None

                # determine label & confidence depending on whether preds returned
                label = "Unknown"
                conf = 0.0
                if preds is not None and len(preds) > 0 and idx is not None and idx < len(preds):
                    out = preds[idx]
                    # out can be shape (2,) or (3,)
                    if out.shape[0] == 2:
                        # binary model: [mask, withoutMask] (older tutorial)
                        mask_prob = float(out[0])
                        nomask_prob = float(out[1])
                        if mask_prob > nomask_prob:
                            label = "Mask"
                            conf = mask_prob
                        else:
                            label = "No Mask"
                            conf = nomask_prob
                    elif out.shape[0] == 3:
                        # 3-class model: assume order [mask, no_mask, incorrect_mask] or similar
                        class_id = int(np.argmax(out))
                        conf = float(out[class_id])
                        if class_id == 0:
                            label = "Mask"
                        elif class_id == 1:
                            label = "No Mask"
                        else:
                            label = "Incorrect Mask"
                    else:
                        # fallback: choose max
                        class_id = int(np.argmax(out))
                        conf = float(out[class_id])
                        label = f"Class_{class_id}"
                else:
                    # fallback: run a single-sample predict (safe but slower)
                    # prepare crop & run the same preprocess as training
                    try:
                        crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        crop_rgb = cv2.resize(crop_rgb, (224,224))
                        x = img_to_array(crop_rgb)
                        x = preprocess_input(x)
                        x = np.expand_dims(x, axis=0)
                        if use_tflite and tflite_interpreter is not None:
                            inp_d = tflite_interpreter.get_input_details()[0]['dtype']
                            qparams = tflite_interpreter.get_input_details()[0].get('quantization', (0.0, 0))
                            scale, zp = qparams
                            if inp_d in [np.int8, np.uint8] and scale != 0:
                                in_data = (x / scale + zp).astype(inp_d)
                            else:
                                in_data = x.astype(inp_d)
                            tflite_interpreter.set_tensor(tflite_interpreter.get_input_details()[0]['index'], in_data)
                            tflite_interpreter.invoke()
                            out = tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[0]['index'])
                            if out.dtype in [np.int8, np.uint8]:
                                o_scale, o_zero = tflite_interpreter.get_output_details()[0].get('quantization', (0.0,0))
                                if o_scale != 0:
                                    out = (out.astype(np.float32) - o_zero) * o_scale
                            out = out[0]
                        else:
                            out = maskNet.predict(x)[0]
                        # interpret as above
                        if out.shape[0] == 2:
                            mask_prob = float(out[0]); nomask_prob = float(out[1])
                            if mask_prob > nomask_prob:
                                label = "Mask"; conf = mask_prob
                            else:
                                label = "No Mask"; conf = nomask_prob
                        elif out.shape[0] == 3:
                            cid = int(np.argmax(out)); conf = float(out[cid])
                            label = ["Mask", "No Mask", "Incorrect Mask"][cid]
                        else:
                            cid = int(np.argmax(out)); conf = float(out[cid]); label = f"Class_{cid}"
                    except Exception:
                        label = "Err"; conf = 0.0

                # update tracker metadata
                tracker.labels[oid] = label
                tracker.last_conf[oid] = conf
                tracker.last_seen[oid] = frame_no
                seen_ids.add(oid)

                # increment counts on first-seen heuristic
                if oid not in counts['_seen_ids']:
                    counts['_seen_ids'].add(oid)
                    if label == "Mask":
                        counts['mask'] += 1
                    elif label == "No Mask":
                        counts['no_mask'] += 1
                    elif label == "Incorrect Mask":
                        counts['incorrect_mask'] += 1

                # draw result
                color = (0,255,0) if label == "Mask" else (0,165,255) if label == "Incorrect Mask" else (0,0,255)
                text = f"ID {oid}: {label} {conf:.2f}"
                cv2.putText(frame, text, (startX, max(15, startY-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                # append CSV log
                ts = datetime.utcnow().isoformat()
                append_log(args['log'], [ts, oid, label, "{:.4f}".format(conf), startX, startY, endX, endY])

                # alert (debounced per object)
                if label != "Mask" and args['alarm'] and os.path.exists(args['alarm']):
                    now = time.time()
                    if last_alert_time.get(oid, 0) + args['debounce'] < now:
                        last_alert_time[oid] = now
                        play_alert(args['alarm'])

            # overlay counts & compliance
            total = len(counts['_seen_ids'])
            m = counts['mask']; nm = counts['no_mask']; im = counts['incorrect_mask']
            comp_pct = (m / total * 100.0) if total > 0 else 0.0
            cv2.putText(frame, f"Total: {total}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Mask:{m} No:{nm} Inc:{im}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Compliance:{comp_pct:.1f}%", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cv2.destroyAllWindows()
        vs.stop()
        print("[INFO] Exiting...")
