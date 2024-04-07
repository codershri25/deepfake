from flask import Flask, render_template, request, redirect, flash
import os
import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
app.secret_key = "secret_key"  # Set a secret key for flashing messages

NUM_FEATURES = 2048

# Function to prepare single video for prediction
def prepare_single_video(frames, feature_extractor, MAX_SEQ_LENGTH, NUM_FEATURES):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

# Function to load video frames
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)

# Function to crop center square of a frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Load pre-trained feature extractor and model
feature_extractor = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
model = load_model(r"D:\DeepFake Detection\Detection\venv\entire_model.h5")

# Function to predict
def predict_video(video_path, feature_extractor, model):
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames, feature_extractor, MAX_SEQ_LENGTH=20, NUM_FEATURES=NUM_FEATURES)
    prediction = model.predict([frame_features, frame_mask])[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            # Ensure the 'uploads' directory exists
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            prediction = predict_video(file_path, feature_extractor, model)
            if prediction >= 0.5:
                result = 'FAKE'
            else:
                result = 'REAL'
            return render_template('result.html', result=result, filename=filename)
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
