import io
import json
import os

import torchvision.models as models
import torchvision.transforms as transforms
import torch
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)       
resnet18 = torch.load('hotdog-resnet18')
resnet18.eval()                                             

# Mapping the output to the classes
class_names = ["It's a Hotdog", 'Not a Hotdog']

# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [transforms.Resize(224),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


# Get a prediction
def get_prediction(input_tensor):
    with torch.no_grad():
        new_pred = resnet18(input_tensor.view(1,3,224,224)).argmax()               # Extract the int value from the PyTorch tensor
    return new_pred.item(), class_names[new_pred.item()]       


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = prediction_idx
            return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()