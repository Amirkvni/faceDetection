from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route("/detect", methods=["POST"])
def detect():
    # دریافت تصویر
    file = request.files['image']
    img_bytes = file.read()
    
    # تبدیل به آرایه NumPy
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # پردازش
    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # بازگشت تصویر به Base64
    _, buffer = cv2.imencode('.jpg', img)
    encoded_img = np.array(buffer).tobytes()

    return encoded_img

@app.route("/", methods=["GET"])
def home():
    return "Face detection API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
