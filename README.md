# Object Detection with YOLO
For this task, I used a pre-trained model in YOLO, that contains a calcultor class in training data.

Python version: 3.10.9
PyTorch version: 2.1.2

## Installation
Install the requirements using the following:
```bash
pip install -r requirements.txt
```

## Usage Instructions
### For run train.py in train (enter train folder first):

Process data with path ./data/images and ./data/labels:
```bash
python script.py --process process_data --data_directory ./data
```
Simple run (if ./data is the data directory):
```bash
python script.py --process process_data
```

Train model:
```bash
python script.py --process train
```
Predict model with model path and path of images folder or image
```bash
python script.py --process predict --model_path <model_path> --images_path <images_path>
```
Simple run:
```bash
python script.py --process predict
```

### Running the Backend Server(after enter folder):
```bash
python .\post_request.py
```
It will start to catch request
