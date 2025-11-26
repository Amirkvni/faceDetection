from flask import Flask, request, jsonify
import cv2
import numpy as np
import time

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route("/detect", methods=["POST"])
def detect():
    start_time = time.time()  # شروع اندازه‌گیری زمان

    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # پردازش چهره
    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    num_faces = len(faces)

    processing_time = (time.time() - start_time) * 1000  # به میلی‌ثانیه

    return jsonify({
        "success": True,
        "num_faces_detected": num_faces,
        "processing_time_ms": round(processing_time, 2)
    })

@app.route("/", methods=["GET"])
def home():
    return "Face detection API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)