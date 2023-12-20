from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
from method import predict_result
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)

model = YOLO('./model/best.pt')
model.fuse()
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"message": "No image part in the request"}), 400

    files = request.files.getlist('image')

    if not files or files[0].filename == '':
        return jsonify({"message": "No image selected for uploading"}), 400

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            # process and predict image
            nparr = np.fromstring(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            prediction = model.predict(source=image)
            results.append(predict_result(prediction, file.filename))

        # test performance
        elif file and allowed_vedio_file(file.filename):
            video_path = os.path.join('videos', secure_filename(file.filename))
            file.save(video_path)

            # start predict vedio
            start_time = time.time()
            prediction = model.predict(source=video_path, stream=True)
            end_time = time.time()
            results.append(predict_result(prediction, file.filename))

            # calculate the time required
            print(f"Start time: {start_time} seconds")
            print(f"End time: {end_time} seconds")
            print(f"Time taken to predict image: {end_time - start_time} seconds")
            os.remove(video_path)
        else:
            return jsonify({"message": "Allowed file types are jpg, jpeg, png"}), 400

    return jsonify({"message": "Image successfully received", "results": results}), 200

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def allowed_vedio_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4','xml'}
if __name__ == "__main__":
    app.run(debug=True)