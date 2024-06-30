from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
from datetime import datetime
import base64

app = Flask(__name__)

# Path to save captured frames
SAVE_PATH = "captured_frames"

# Number of classes
#num_classes = 6

# Initialize variables
capturing = False
current_class = 0
cap = None

def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")

def release_camera():
    global cap
    if cap is not None:
        cap.release()

def create_class_folder(class_idx):
    save_folder = os.path.join(SAVE_PATH, f"class{class_idx}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

def save_frame(frame, class_idx):
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    save_folder = os.path.join(SAVE_PATH, f"class{class_idx}")
    cv2.imwrite(os.path.join(save_folder, filename), frame)

# def save_frame1(frame, class_idx):
#     filename = f"1.jpg"
#     save_folder = os.path.join("static", f"Image")
#     cv2.imwrite(os.path.join(save_folder, filename), frame)

def predict(data):
    # Perform prediction based on the input data
    # For example, if you receive an image, use a trained model to predict the class
    # Here, I'm assuming the prediction is based on some integer data
    # You should replace this with your actual prediction logic
    # For demonstration, I'm returning a hardcoded prediction
    return 0

def gen_frames():
    global capturing, current_class, cap
    initialize_camera()
    while True:
        success, frame = cap.read()
        if capturing:
            if success:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg_as_text.encode('utf-8') + b'\r\n')
                    save_frame(frame, current_class)
        else:
            print(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            # save_frame1(frame, current_class)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Assuming you receive data needed for prediction in the request body
    data = request.json
    # Perform prediction
    prediction = predict(data)
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})


@app.route('/toggle_capture', methods=['POST'])
def toggle_capture():
    global capturing, current_class

    capturing = not capturing

    if capturing:
        start_time = datetime.now()
        create_class_folder(current_class)
        return jsonify({"message": "Capture started."})
    else:
        #release_camera()
        current_class = current_class + 1
        return jsonify({"message": f"Switched to class {current_class}."})

if __name__ == '__main__':
    app.run(debug=True)
