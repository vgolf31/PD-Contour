from flask import Flask, request, render_template, jsonify, send_file
import torch
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
import time  # Import the time module for adding delays

app = Flask(__name__)

# Load the CNN model for prediction
model = torch.load('/Users/veer/Desktop/HackRU/parkinsons_vgg_model_scripted.pth', map_location=torch.device('cpu'))
model.eval()

# Define image transformations for the prediction model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Constants for mask creation
MIN_CONTOUR_AREA = 0
MAX_CONTOUR_AREA = 250
EDGE_MARGIN = 80

def create_substantia_nigra_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    height, width = image.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA) and \
           (x > EDGE_MARGIN and y > EDGE_MARGIN and 
            x + w < width - EDGE_MARGIN and y + h < height - EDGE_MARGIN):
            cv2.drawContours(mask, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
    return mask

def overlay_mask(image, mask):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)
    return overlay

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_stream = io.BytesIO(file.read())
    image = Image.open(image_stream).convert("RGB")
    image = transform(image).unsqueeze(0)

    time.sleep(0.25)  # Simulate a delay of 3 seconds

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    prediction_label = "Parkinson's" if predicted_class.item() == 1 else "No Parkinson's"
    return jsonify({'prediction': prediction_label})

@app.route('/mask', methods=['POST'])
def mask():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_stream = io.BytesIO(file.read())
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    time.sleep(1)  # Simulate a delay of 3 seconds

    mask = create_substantia_nigra_mask(image)
    overlay_image = overlay_mask(image, mask)

    _, buffer = cv2.imencode('.png', overlay_image)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
