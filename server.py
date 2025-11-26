from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return "No image provided", 400

    file = request.files['image']
    img_bytes = file.read()
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image", 400

    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ذخیره در حافظه و ارسال
    _, buffer = cv2.imencode('.jpg', img)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

@app.route("/", methods=["GET"])
def home():
    return "Cloud Face Detection API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)