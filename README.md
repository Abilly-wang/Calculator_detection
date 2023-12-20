For this task, I used a pre-trained model in YOLO, that contains a calcultor class in training data.

Install the requirements using the following:
pip install -r requirements.txt

For run train.py in train (enter train folder first):

Process data with path ./data/images and ./data/labels:
    python script.py --process process_data --data_directory ./data
    Simple run (if ./data is the data directory)::
        python script.py --process process_data

Train model:
python script.py --process train

Predict model with model path and path of images folder or image
    python script.py --process predict --model_path <model_path> --images_path <images_path>
    Simple run:
        python script.py --process predict

For run backend(after enter folder):
python .\post_request.py
It will start to catch request
