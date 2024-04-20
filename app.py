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
start_time = None
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

def gen_frames():
    global capturing, current_class, start_time, cap

    while True:
        if capturing and (datetime.now() - start_time).total_seconds() < 50:
            if cap is None or not cap.isOpened():
                initialize_camera()
            success, frame = cap.read()
            if success:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg_as_text.encode('utf-8') + b'\r\n')
                    save_frame(frame, current_class)
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_capture', methods=['POST'])
def toggle_capture():
    global capturing, current_class, start_time

    capturing = not capturing

    if capturing:
        start_time = datetime.now()
        create_class_folder(current_class)
        return jsonify({"message": "Capture started."})
    else:
        release_camera()
        current_class = current_class + 1
        return jsonify({"message": f"Switched to class {current_class}."})

@app.route('/exit', methods=['GET'])
def exit_app():
    release_camera()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    else:
        func()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(debug=True)
