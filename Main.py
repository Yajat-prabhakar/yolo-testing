import sys
import cv2
import math
import queue
import json
import threading
import numpy as np
import pyttsx3
import sounddevice as sd
from ultralytics import YOLO
from vosk import Model, KaldiRecognizer
import requests
import os
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QFileDialog, QFrame, QSizePolicy, QStackedLayout, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt

def get_medical_help_from_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', 'your-api-key-here')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a medical assistant in space helping with astronaut emergencies."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except RuntimeError as e:
        print("TTS Error:", e)

q = queue.Queue()
vosk_model = Model("model/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(vosk_model, 16000)

def audio_callback(indata, frames, time, status):
    q.put(bytes(indata))

class VoiceWorker(QObject):
    result_ready = pyqtSignal(str)
    def start_listening(self):
        def run():
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                   channels=1, callback=audio_callback):
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "")
                        self.result_ready.emit(text)
                        break
        threading.Thread(target=run, daemon=True).start()

def is_unconscious(keypoints):
    try:
        nose = keypoints[0]
        l_shoulder, r_shoulder = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]

        shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
        hip_mid = np.mean([l_hip, r_hip], axis=0)
        torso_angle = math.degrees(math.atan2(hip_mid[1] - shoulder_mid[1], hip_mid[0] - shoulder_mid[0]))

        head_slumped = (nose[1] - shoulder_mid[1]) > 40
        is_flat = abs(torso_angle) < 25
        shoulders_level = abs(l_shoulder[1] - r_shoulder[1]) < 20

        return head_slumped and is_flat and shoulders_level, {
            "head_slumped": head_slumped,
            "torso_flat": is_flat,
            "shoulders_level": shoulders_level,
            "torso_angle": torso_angle
        }
    except Exception as e:
        return False, {"error": str(e)}

def head_tilt_angle(keypoints):
    try:
        nose, l_eye, r_eye = keypoints[0], keypoints[1], keypoints[2]
        def angle(p1, p2):
            return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return abs((angle(nose, l_eye) + angle(nose, r_eye)) / 2)
    except:
        return 0

pose_model = YOLO("yolov8s-pose.pt")
pretrained_model = YOLO("yolov8n.pt")
custom_model = YOLO(r"C:\\Users\\Yajat\\runs\\detect\\train\\weights\\best.pt")

class AssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛰️ Space Station Vision Assistant")
        self.setStyleSheet("background-color: #1e1e2f; color: #ffffff;")
        self.resize(1100, 700)

        self.label = QLabel("🛰️ Speak, Upload, or Monitor Live Feed")
        self.label.setFont(QFont("Orbitron", 18, QFont.Bold))
        self.label.setStyleSheet("color: #00ffcc; margin: 10px;")
        self.label.setAlignment(Qt.AlignCenter)

        self.textbox = QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setMinimumHeight(140)
        self.textbox.setStyleSheet("background-color: #2c2f4a; color: #00ff9f; padding: 10px; border: 1px solid #00ffcc; border-radius: 6px;")

        self.listen_button = QPushButton("🎙️ Start Listening")
        self.listen_button.setStyleSheet("background-color: #00d46a; color: white; padding: 12px; font-size: 14px; font-weight: bold; border-radius: 8px;")
        self.listen_button.clicked.connect(self.listen)

        self.upload_button = QPushButton("📁 Upload Image")
        self.upload_button.setStyleSheet("background-color: #1f6feb; color: white; padding: 12px; font-size: 14px; font-weight: bold; border-radius: 8px;")
        self.upload_button.clicked.connect(self.upload_image)

        self.camLabel = QLabel()
        self.camLabel.setFixedHeight(400)
        self.camLabel.setStyleSheet("border: 2px solid #3f3f5f; border-radius: 8px; background-color: #12121c;")
        self.camLabel.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        button_row = QHBoxLayout()
        button_row.setSpacing(20)
        button_row.addWidget(self.listen_button)
        button_row.addWidget(self.upload_button)

        self.layout.addWidget(self.label)
        self.layout.addLayout(button_row)
        self.layout.addWidget(self.camLabel)
        self.layout.addWidget(self.textbox)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        self.voice_worker = VoiceWorker()
        self.voice_worker.result_ready.connect(self.handle_voice_input)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        results_pose = pose_model(frame)[0]
        annotated = results_pose.plot()

        for pose in results_pose.keypoints:
            keypoints = pose.data[0].cpu().numpy()
            if len(keypoints) < 13 or np.isnan(keypoints).any():
                continue
            xy = keypoints[:, :2]
            unconscious, debug_info = is_unconscious(xy)
            tilt = head_tilt_angle(xy)

            if tilt > 25 and debug_info.get("head_slumped"):
                unconscious = True

            label = "⚠ UNCONSCIOUS" if unconscious else "Conscious"
            color = (0, 0, 255) if unconscious else (0, 255, 0)

            min_x, min_y = int(np.min(xy[:, 0])), int(np.min(xy[:, 1]))
            max_x, max_y = int(np.max(xy[:, 0])), int(np.max(xy[:, 1]))
            cv2.rectangle(annotated, (min_x, min_y), (max_x, max_y), color, 2)
            cv2.putText(annotated, label, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        img = QImage(annotated.data, annotated.shape[1], annotated.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.camLabel.setPixmap(QPixmap.fromImage(img))

    def listen(self):
        self.label.setText("🎤 Listening... Speak now")
        self.voice_worker.start_listening()

    def handle_voice_input(self, text):
        self.label.setText(f"You said: {text}")
        self.textbox.append(f"> {text}")
        response = self.get_response(text.lower())
        speak(response)
        self.textbox.append(f"Bot: {response}")

    def get_response(self, text):
        if "hello" in text:
            return "Hello, astronaut. Ready to assist."
        elif "object" in text or "detect" in text:
            return self.detect_from_camera()
        elif "stop" in text or "exit" in text:
            speak("Shutting down.")
            sys.exit()
        else:
            return "Command not recognized."

    def detect_from_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return "Camera unavailable."
        results_c = custom_model(frame)[0]
        results_p = pretrained_model(frame)[0]
        detected = self.merge_labels(results_c, results_p)
        if detected:
            message = f"Detected: {', '.join(detected)}"
            self.textbox.append(f"🧠 {message}")
            speak(message)
            return message
        else:
            return "No objects detected."

    def upload_image(self):
        def is_unconscious(keypoints, tilt_angle):
            try:
                nose = keypoints[0]
                l_shoulder, r_shoulder = keypoints[5], keypoints[6]
                l_hip, r_hip = keypoints[11], keypoints[12]

                shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
                hip_mid = np.mean([l_hip, r_hip], axis=0)

                vertical_shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
                torso_angle = math.degrees(math.atan2(
                    hip_mid[1] - shoulder_mid[1], hip_mid[0] - shoulder_mid[0]
                ))

                torso_flat = 75 < abs(torso_angle) < 105
                shoulders_level = vertical_shoulder_diff < 20
                head_slumped = (nose[1] - shoulder_mid[1]) > 40 or tilt_angle > 45

                is_unconscious = head_slumped and torso_flat and shoulders_level

                return is_unconscious, {
                    "head_slumped": head_slumped,
                    "torso_flat": torso_flat,
                    "shoulders_level": shoulders_level,
                    "torso_angle": torso_angle,
                    "tilt_triggered": tilt_angle > 45
                }

            except Exception as e:
                return False, {"error": str(e)}

        path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        img = cv2.imread(path)

        results_c = custom_model(img)[0]
        results_p = pretrained_model(img)[0]
        results_pose = pose_model(img)[0]

        img1 = results_c.plot()
        img2 = results_p.plot()

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        combined = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        for pose in results_pose.keypoints:
            keypoints = pose.data[0].cpu().numpy()
            if len(keypoints) < 13 or np.isnan(keypoints).any():
                continue
            xy = keypoints[:, :2]

            def angle(p1, p2):
                return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

            tilt = abs((angle(xy[0], xy[1]) + angle(xy[0], xy[2])) / 2)
            unconscious, debug = is_unconscious(xy, tilt)

            label = "⚠ UNCONSCIOUS" if unconscious else "Conscious"
            color = (0, 0, 255) if unconscious else (0, 255, 0)

            min_x, min_y = int(np.min(xy[:, 0])), int(np.min(xy[:, 1]))
            max_x, max_y = int(np.max(xy[:, 0])), int(np.max(xy[:, 1]))

            cv2.rectangle(combined, (min_x, min_y), (max_x, max_y), color, 2)
            cv2.putText(combined, label, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            self.textbox.append(f"🧍 Person: {label} - Tilt: {tilt:.1f}° | Debug: {debug}")

            # If unconscious, ask Groq for help
            if unconscious:
                symptoms = f"The astronaut is unconscious. Head slumped: {debug['head_slumped']}, torso flat: {debug['torso_flat']}, shoulders level: {debug['shoulders_level']}, torso angle: {debug['torso_angle']:.2f}°."
                help_prompt = symptoms + " What could be wrong and what should I do to help them?"
                self.textbox.append("🤖 Asking medical assistant for help...")
                response = get_medical_help_from_groq(help_prompt)
                self.textbox.append(f"🆘 Groq AI: {response}")
                speak(response)  # TTS

            self.textbox.append(f"🧍 Person: {label} - Tilt: {tilt:.1f}° | Debug: {debug}")

        qimg = QImage(combined.data, combined.shape[1], combined.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.camLabel.setPixmap(QPixmap.fromImage(qimg))

        detected = self.merge_labels(results_c, results_p)
        result = f"In the uploaded image, I see: {', '.join(detected)}." if detected else "No objects detected."
        self.textbox.append("> Uploaded Image")
        self.textbox.append(f"Bot: {result}")
        speak(result)


    def merge_labels(self, results_custom, results_pre, threshold=0.4):
        names_c = results_custom.names
        names_p = results_pre.names
        det_c = [names_c[int(cls)] for cls, conf in zip(results_custom.boxes.cls, results_custom.boxes.conf) if conf > threshold]
        det_p = [names_p[int(cls)] for cls, conf in zip(results_pre.boxes.cls, results_pre.boxes.conf) if conf > threshold]
        return list(set(det_c + det_p))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AssistantApp()
    window.show()
    sys.exit(app.exec_())