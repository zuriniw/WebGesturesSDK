# flask_app.py
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from eyeGestures import EyeGestures_v2
from threading import Thread 
import numpy as np
import base64
import cv2

class KalmanFilter:
    def __init__(self, process_variance=0.1, measurement_variance=0.1, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = 1.0
        
    def update(self, measurement):
        # Prediction
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

flask_app = Flask(__name__)
socketio = SocketIO(flask_app)

gestures = EyeGestures_v2()

# Initialize Kalman filters for x and y coordinates
kf_x = KalmanFilter(process_variance=0.1, measurement_variance=1.0)
kf_y = KalmanFilter(process_variance=0.1, measurement_variance=1.0)

# Add a velocity-based threshold to detect and prevent sticking
last_x = None
last_y = None
last_time = None
stick_threshold = 0.1  # pixels per millisecond

# Serve the HTML page with JavaScript for WebSocket communication

@flask_app.route('/')
def index():
    return render_template('v2_test.html')


@flask_app.route('/eyeCanvas.js')
def eyecanvas(): 
    # unique_id = tasks.generater_ID()
    return render_template(
        'sdk/eyeCanvas.js',
        domain=request.host,
        unique_id="unique_id")

def base64cv2(img):
    image_data = img.split(',')[1]  # Remove data URI header
    image_array = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return frame

def check_sticking(x, y, current_time):
    global last_x, last_y, last_time
    
    if last_x is None or last_y is None or last_time is None:
        last_x, last_y, last_time = x, y, current_time
        return False
    
    time_diff = current_time - last_time
    if time_diff == 0:
        return False
        
    velocity = np.sqrt((x - last_x)**2 + (y - last_y)**2) / time_diff
    
    last_x, last_y, last_time = x, y, current_time
    
    return velocity < stick_threshold

# Handle WebSocket connections
@socketio.on('msg_data')
def on_stream(data):
    calibrate = data['calibrate']
    frame = base64cv2(data['image'])
    current_time = data.get('timestamp', 0)

    try:
        event, calibration = gestures.step(frame, calibrate, data['width'], data['height'])
        
        # Apply Kalman filtering to smooth the coordinates
        raw_x, raw_y = event.point[0], event.point[1]
        
        # Check for sticking
        if check_sticking(raw_x, raw_y, current_time):
            # If sticking detected, increase process variance temporarily
            kf_x.process_variance = 0.5
            kf_y.process_variance = 0.5
        else:
            # Reset process variance to normal
            kf_x.process_variance = 0.1
            kf_y.process_variance = 0.1
        
        # Apply Kalman filter
        smoothed_x = kf_x.update(raw_x)
        smoothed_y = kf_y.update(raw_y)
        
        emit('rsp', {
            "x": smoothed_x,
            "y": smoothed_y,
            "c_x": calibration.point[0],
            "c_y": calibration.point[1]
        })
    except Exception as e:
        print(f"Caught expression: {str(e)}")
        emit('rsp', {"x": 0, "y": 0, "c_x": 0, "c_y": 0})

if __name__ == '__main__':
    socketio.run(app=flask_app,host="localhost",port=8000)
