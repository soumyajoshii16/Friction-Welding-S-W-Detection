from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import os
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from datetime import datetime
import subprocess
import sys
import webbrowser

app = Flask(__name__)
uploaded_frames_dir = 'webcam_frames'
prediction_file = 'prediction_result.txt'

# Load the model
num_classes = 3
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('fasterrcnn_model8.pth', map_location=torch.device('cpu')))
device = torch.device('cpu')
model.to(device)
model.eval()

# Function to preprocess frames
def preprocess(frame):
    pil_image = Image.fromarray(frame)
    transform = T.ToTensor()
    return transform(pil_image).unsqueeze(0)

# Function to draw bounding boxes and labels
def draw_boxes(frame, predictions, threshold=0.5):
    labels_map = {1: 'Green', 2: 'White'}
    detected_colors = []
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score >= threshold:
            box = box.cpu().numpy().astype(int)
            label = label.item()
            color = (0, 255, 0) if label == 1 else (255, 255, 255) if label == 2 else (0, 0, 0)
            detected_colors.append(label)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, labels_map[label], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return detected_colors

# Function to predict validity based on wheel type and detected colors
def model_predict(wheel_type, detected_colors):
    if wheel_type == '328' and 1 in detected_colors:
        return "Valid Shaft"
    elif wheel_type == '165' and 2 in detected_colors:
        return "Valid Shaft"
    else:
        return "Invalid Shaft"

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise Exception("Could not open video device")
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        return image

camera = VideoCamera()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        wheel_type = request.form.get('wheel_type')
        frame, detected_colors = capture_and_predict()
        if frame is None:
            return "Failed to capture image", 500
        
        # Save the captured frame
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #frame_path = os.path.join(uploaded_frames_dir, f'{wheel_type}_frame_{timestamp}.jpg')
        #cv2.imwrite(frame_path, frame)
        
        # Determine shaft validity based on wheel type and detected colors
        validity_text = model_predict(wheel_type, detected_colors)
        
        # Convert validity to 1 or 0 and save to the prediction file
        validity_bit = '1' if validity_text == "Valid Shaft" else '0'
        with open(prediction_file, 'w') as file:
            file.write(validity_bit)
        
        return render_template('result.html', wheel_type=wheel_type, validity_text=validity_text)
    
    return render_template('result.html')

def capture_and_predict():
    frame = camera.get_frame()
    if frame is None:
        return None, "Failed to capture image"
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_image = preprocess(frame_rgb).to(device)
    with torch.no_grad():
        prediction = model(transformed_image)
    detected_colors = draw_boxes(frame, prediction)
    return frame, detected_colors

@app.route('/prediction_result.txt', methods=['GET', 'POST'])
def handle_prediction_file():
    if request.method == 'GET':
        try:
            with open(prediction_file, 'r') as file:
                content = file.read().strip()
            return content, 200
        except FileNotFoundError:
            return "File not found", 404
    elif request.method == 'POST':
        clear = request.form.get('clear')
        if clear == 'true':
            try:
                with open(prediction_file, 'w') as file:
                    file.write('')
                return "Prediction file cleared successfully", 200
            except Exception as e:
                return f"Failed to clear prediction file: {str(e)}", 500
        else:
            return "Invalid request: 'clear' parameter must be 'true'", 400

def start_http_server():
    try:
        subprocess.Popen([sys.executable, "-m", "http.server", "8000"])
        
        # Open the browser automatically with your Flask app's URL
        webbrowser.open('http://127.0.0.1:5000')
    except Exception as e:
        app.logger.error(f"Failed to start http.server: {e}")

if __name__ == '__main__':
    if not os.path.exists(uploaded_frames_dir):
        os.makedirs(uploaded_frames_dir)
    
    start_http_server()
    app.run(debug=True, use_reloader=False)