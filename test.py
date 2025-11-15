from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import joblib
from scipy.spatial import distance as dist
import time
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)

# ========================
# Load Trained Model & Scaler
# ========================
model = joblib.load("driver_drowsiness_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# Initialize MediaPipe FaceMesh
# ========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# ========================
# EAR & MAR Calculation Functions
# ========================
def eye_aspect_ratio(landmarks):
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]
    left = np.array([landmarks[i] for i in left_eye])
    right = np.array([landmarks[i] for i in right_eye])

    def ear(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    return (ear(left) + ear(right)) / 2.0


def mouth_aspect_ratio(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[78]
    right_mouth = landmarks[308]
    return dist.euclidean(top_lip, bottom_lip) / dist.euclidean(left_mouth, right_mouth)

# ========================
# Parameters
# ========================
EAR_THRESHOLD_FACTOR = 0.82
MAR_THRESHOLD = 0.50
FRAME_WINDOW = 10
SUSTAIN_SECONDS = 2
FPS_EST = 15
FRAME_LIMIT = max(4, int(SUSTAIN_SECONDS * FPS_EST))

last_label = "Driver Active (0.0%)"

# ========================
# Load Custom Font
# ========================
# Make sure this font path exists (Algerian font or replace with Poppins/Roboto/etc.)
FONT_PATH = "C:/Windows/Fonts/ALGER.TTF"  # On Windows
font = ImageFont.truetype(FONT_PATH, 38)  # Font size

# ========================
# Frame Generation
# ========================
def gen_frames():
    global last_label
    cap = cv2.VideoCapture(0)
    time_start = time.time()
    open_ears = []
    calibration_done = False
    EAR_values = []
    MAR_values = []
    closed_counter = 0
    yawn_counter = 0

    print("✅ Camera Stream Started")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        label = "Driver Active (0.0%)"
        color = (0, 255, 0)

        # ================= Calibration =================
        if not calibration_done:
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
                    try:
                        ear = eye_aspect_ratio(landmarks)
                        open_ears.append(ear)
                    except:
                        pass

            if time.time() - time_start > 2.5 and len(open_ears) > 0:
                calibrated_open = np.mean(open_ears)
                EAR_THRESHOLD = calibrated_open * EAR_THRESHOLD_FACTOR
                calibration_done = True
                print(f"✅ Calibration Done. EAR_THRESHOLD = {EAR_THRESHOLD:.3f}")
            else:
                cv2.rectangle(frame, (0, 0), (640, 70), (20, 20, 20), -1)
                cv2.putText(frame, "Calibrating... please look at camera",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue
        else:
            EAR_THRESHOLD = np.mean(open_ears) * EAR_THRESHOLD_FACTOR

        # ================= Detection =================
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
                try:
                    ear = eye_aspect_ratio(landmarks)
                    mar = mouth_aspect_ratio(landmarks)
                except:
                    continue

                EAR_values.append(ear)
                MAR_values.append(mar)
                if len(EAR_values) > FRAME_WINDOW:
                    EAR_values.pop(0)
                    MAR_values.pop(0)

                EAR_mean = float(np.mean(EAR_values))
                EAR_std = float(np.std(EAR_values))
                MAR_mean = float(np.mean(MAR_values))
                MAR_std = float(np.std(MAR_values))

                X_test = np.array([[EAR_mean, EAR_std, MAR_mean, MAR_std]])
                try:
                    X_scaled = scaler.transform(X_test)
                    prob = model.predict_proba(X_scaled)[0]
                    pred = int(np.argmax(prob))
                    confidence = float(np.max(prob)) * 100
                except:
                    pred = 0
                    confidence = 0.0

                closed_counter = closed_counter + 1 if EAR_mean < EAR_THRESHOLD else max(0, closed_counter - 1)
                yawn_counter = yawn_counter + 1 if MAR_mean > MAR_THRESHOLD else max(0, yawn_counter - 1)

                if closed_counter >= FRAME_LIMIT:
                    label = f"⚠️ Eyes Closed - Drowsy ({confidence:.1f}%)"
                    color = (255, 60, 60)
                elif yawn_counter >= FRAME_LIMIT:
                    label = f"⚠️ Yawning - Drowsy ({confidence:.1f}%)"
                    color = (255, 165, 0)
                elif pred == 1 and confidence > 60:
                    label = f"⚠️ Drowsy ({confidence:.1f}%)"
                    color = (255, 50, 50)
                else:
                    label = f"Driver Active ({confidence:.1f}%)"
                    color = (50, 255, 100)

                last_label = label

        # ================= Draw Text using PIL =================
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw the label with custom font
        draw.text((30, 20), last_label.upper(), font=font, fill=color)

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ========================
# Flask Routes
# ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/label')
def label():
    return jsonify({"label": last_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
