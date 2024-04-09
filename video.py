import datetime
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image
import io
import cv2
import pickle
from fer import FER
import os

app = Flask(__name__)

# Load known face encodings from JPEG images
def load_known_face_encodings():
    known_face_encodings = []

    # Assuming face images are stored in a directory named 'known_faces'
    known_faces_dir = 'known_faces'

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            face_encoding = face_recognition.face_encodings(face_image)[0]  # Assuming only one face in each image
            known_face_encodings.append(face_encoding)

    return known_face_encodings

known_face_encodings = load_known_face_encodings()

def recognize_person(frame, known_face_encodings):
    small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(small_frame, model="hog")

    face_names = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = small_frame[top:bottom, left:right]
        face_encodings = face_recognition.face_encodings(face_image)
        print("outer loop")

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown" if not matches else "Person {}".format(matches.index(True) + 1)
            face_names.append(name)
            print("inner loop")

    return face_names

def detect_emotion(frame):
    detector = FER()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    emotions_list = detector.detect_emotions(frame_bgr)

    dominant_emotions = []
    for emotions in emotions_list:
        if emotions:
            dominant_emotion = max(emotions['emotions'].items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "Unknown"

        dominant_emotions.append(dominant_emotion)

    return dominant_emotions

@app.route('/process_frame', methods=['POST'])
def process_frame():
    image_bytes = request.data
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)

    face_names = recognize_person(frame, known_face_encodings)
    emotions = detect_emotion(frame)

    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "Unknown"

    response = {
        #'detected_faces': face_names,
        'dominant_emotion': dominant_emotion
    }
    print(dominant_emotion)

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
