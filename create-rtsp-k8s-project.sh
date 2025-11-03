#!/bin/bash

# This script creates a new, clean directory for your project.
# Run it with: bash create_rtsp_k8s_project.sh

PROJECT_DIR="rtsp-k8s-project"
mkdir -p $PROJECT_DIR/templates
mkdir -p $PROJECT_DIR/static

echo "--- Created directory $PROJECT_DIR/ ---"

# --- 1. README.md (NEW INSTRUCTIONS!) ---
cat <<'EOF' > $PROJECT_DIR/README.md
# RTSP Object Detector - v2 Deployment Guide

This project is now configured with InfluxDB history. Follow these "cache-busting" steps **exactly** to deploy the new version and guarantee your cluster runs the new code.

## 1. Add Placeholder Image

- **Manually** add your `camera_unavailable.jpg` file to the `static/` folder.

## 2. Update Placeholders

You **MUST** edit these files and replace the placeholders:

1.  **`k8s-influxdb.yaml`**:
    - `YOUR_INFLUXDB_TOKEN`: Set a strong, secure token here.

2.  **`k8s-manifest.yaml`**:
    - `YOUR_AZURE_KEY_HERE`: Set your Azure key.
    - `https.your-endpoint.cognitiveservices.azure.com/`: Set your Azure endpoint.
    - `rtsp://your-stream-url-here`: Set your RTSP stream URL.
    - `YOUR_INFLUXDB_TOKEN`: Use the *same token* from step 1.
    - `http://rtsp-detector.your-domain.com`: Set your public URL.
    - `your-registry/rtsp-object-detector:v1`: **IMPORTANT!** Update the `your-registry` part to your Docker registry (e.g., `docker.io/yourusername`).

## 3. Build & Push the **Versioned** Image

This is the most critical step. We are using a `v1` tag, not `:latest`.

```bash
# Navigate to this project directory
cd rtsp-k8s-project-v2

# 1. Build the image with a NEW tag
docker build -t your-registry/rtsp-object-detector:v1 .

# 2. Push the image
docker push your-registry/rtsp-object-detector:v1
```

## 4. Deploy to Kubernetes

```bash
# 1. Deploy InfluxDB first and wait for it to be 'Running'
kubectl apply -f k8s-influxdb.yaml
kubectl get pods -w

# 2. Deploy the application
kubectl apply -f k8s-manifest.yaml
```

## 5. Verify

Because you used a new tag (`:v1`), Kubernetes is **forced** to pull the new image.

- Check the logs of the `web-app` container. You should see it initialize the InfluxDB client.
- Open your browser and do a **hard refresh** (`Ctrl+Shift+R` or `Cmd+Shift+R`).
- The "Analyze" and "History" buttons should now work.

## Updating in the Future

If you make new code changes, build and push with a **new** tag (e.g., `:v2`), update the `image:` line in `k8s-manifest.yaml`, and run `kubectl apply` again.
EOF

echo "--- Created README.md ---"

# --- 2. Dockerfile ---
cat <<'EOF' > $PROJECT_DIR/Dockerfile
# Use a modern, supported Python base image (Debian 11 "Bullseye")
FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app

# Install OS-level dependencies
# - ffmpeg & gstreamer: for OpenCV RTSP stream handling
# - libsm6 & libxext6: required by OpenCV GUI (even if headless)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 8000

# Note: The 'CMD' is now in the k8s manifest (as 'command' and 'args')
# This makes the image more flexible.
EOF

echo "--- Created Dockerfile ---"

# --- 3. requirements.txt ---
cat <<'EOF' > $PROJECT_DIR/requirements.txt
Flask
gunicorn
gevent
opencv-python-headless
numpy
azure-cognitiveservices-vision-computervision
msrest
influxdb-client
EOF

echo "--- Created requirements.txt ---"

# --- 4. k8s-influxdb.yaml ---
cat <<'EOF' > $PROJECT_DIR/k8s-influxdb.yaml
# --- InfluxDB v2 Deployment ---
# This manifest creates a standalone InfluxDB pod with persistent storage.
# It uses the 'local-path' provisioner available in RKE2.

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: influxdb-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path # Uses RKE2's default provisioner
  resources:
    requests:
      storage: 5Gi # Allocate 5GB for the database
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: influxdb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: influxdb
  template:
    metadata:
      labels:
        app: influxdb
    spec:
      containers:
        - name: influxdb
          image: influxdb:2.7.5-alpine
          ports:
            - containerPort: 8086
          env:
            - name: DOCKER_INFLUXDB_INIT_MODE
              value: "setup"
            - name: DOCKER_INFLUXDB_INIT_USERNAME
              value: "admin"
            - name: DOCKER_INFLUXDB_INIT_PASSWORD
              value: "adminpassword" # Not used for token auth, but good to set
            - name: DOCKER_INFLUXDB_INIT_ORG
              value: "rtsp-project-org"
            - name: DOCKER_INFLUXDB_INIT_BUCKET
              value: "analysis_data"
            - name: DOCKER_INFLUXDB_INIT_ADMIN_TOKEN
              value: "YOUR_INFLUXDB_TOKEN" # !!! REPLACE THIS WITH YOUR SECURE TOKEN !!!
          volumeMounts:
            - name: influxdb-storage
              mountPath: /var/lib/influxdb2
      volumes:
        - name: influxdb-storage
          persistentVolumeClaim:
            claimName: influxdb-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: influxdb-service
spec:
  selector:
    app: influxdb
  ports:
    - protocol: TCP
      port: 8086
      targetPort: 8086
EOF

echo "--- Created k8s-influxdb.yaml ---"

# --- 5. k8s-manifest.yaml (Main App) ---
cat <<'EOF' > $PROJECT_DIR/k8s-manifest.yaml
# This manifest contains all resources for the main application.
# It uses a single pod with two containers (web-app, camera-manager)
# that communicate via a shared in-memory volume.

# --- 1. Secret ---
# Holds all confidential data
apiVersion: v1
kind: Secret
metadata:
  name: rtsp-config
type: Opaque
stringData:
  # --- Azure Credentials ---
  AZURE_VISION_SUBSCRIPTION_KEY: "YOUR_AZURE_KEY_HERE"
  AZURE_VISION_ENDPOINT: "https://your-endpoint.cognitiveservices.azure.com/"
  
  # --- Camera Configuration ---
  CAMERA_SOURCE: "rtsp"
  RTSP_STREAM_URL: "rtsp://your-stream-url-here"
  
  # --- InfluxDB Connection Details (MUST MATCH k8s-influxdb.yaml) ---
  INFLUXDB_URL: "http://influxdb-service:8086"
  INFLUXDB_TOKEN: "YOUR_INFLUXDB_TOKEN" # !!! REPLACE THIS WITH THE SAME TOKEN !!!
  INFLUXDB_ORG: "rtsp-project-org"
  INFLUXDB_BUCKET: "analysis_data"

---
# --- 2. Deployment ---
# Deploys the application pod
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtsp-object-detector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rtsp-object-detector
  template:
    metadata:
      labels:
        app: rtsp-object-detector
    spec:
      volumes:
        # 1. Shared memory volume for IPC between containers
        - name: shared-frame-memory
          emptyDir:
            medium: Memory # Use RAM for high-speed IPC
            sizeLimit: 128Mi
        # 2. Config file for camera switching
        - name: camera-config-volume
          emptyDir: {}
      
      containers:
        # --- Container 1: The Python/Flask Web App ---
        - name: web-app
          # !!! REPLACE THIS with your new versioned image !!!
          image: "your-registry/rtsp-object-detector:v1"
          imagePullPolicy: Always # Ensures the new image is pulled
          ports:
            - containerPort: 8000
          
          # Command to run Gunicorn
          command: ["gunicorn"]
          args: [
            "--workers", "3",
            "--worker-class", "gevent",
            "--bind", "0.0.0.0:8000",
            "--log-level", "info",
            "app:app"
          ]
          
          envFrom:
            - secretRef:
                name: rtsp-config # Mounts all secrets as env variables
          
          volumeMounts:
            - name: shared-frame-memory
              mountPath: /dev/shm
            - name: camera-config-volume
              mountPath: /app/config

        # --- Container 2: The OpenCV Camera Manager ---
        - name: camera-manager
          # !!! REPLACE THIS with your new versioned image !!!
          image: "your-registry/rtsp-object-detector:v1"
          imagePullPolicy: Always # Ensures the new image is pulled
          
          # Command to run the camera manager script
          command: ["python3"]
          args: ["camera_manager.py"]
          
          envFrom:
            - secretRef:
                name: rtsp-config # Mounts all secrets as env variables
          
          volumeMounts:
            - name: shared-frame-memory
              mountPath: /dev/shm
            - name: camera-config-volume
              mountPath: /app/config

---
# --- 3. Service ---
# Exposes the web-app container inside the cluster
apiVersion: v1
kind: Service
metadata:
  name: rtsp-detector-service
spec:
  selector:
    app: rtsp-object-detector
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000

---
# --- 4. Ingress ---
# Exposes the service to the internet via your Nginx Ingress Controller
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rtsp-detector-ingress
spec:
  rules:
    # !!! REPLACE THIS with your public domain !!!
    - host: "rtsp-detector.your-domain.com"
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: rtsp-detector-service
                port:
                  number: 80
EOF

echo "--- Created k8s-manifest.yaml ---"

# --- 6. app.py (The Web App) ---
cat <<'EOF' > $PROJECT_DIR/app.py
import os
import time
import json
import logging
import multiprocessing.shared_memory as shared_memory
import cv2
import numpy as np
import base64
import io

from flask import Flask, render_template, Response, jsonify, request
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import datetime as dt # Use a distinct alias for timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('web_app')

app = Flask(__name__)

# --- Frame Configuration (Must match camera_manager.py) ---
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
FRAME_CHANNELS = 3
SHM_NAME = 'object_detector_frame_buffer'
SHM_SIZE = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS

# --- Analysis Config ---
MIN_OBJECT_CONFIDENCE = 0.60  # Minimum confidence (0.0 to 1.0) required for display

# --- Environment Variables ---
AZURE_VISION_KEY = os.environ.get("AZURE_VISION_SUBSCRIPTION_KEY")
AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT")

# --- InfluxDB Config (From k8s-manifest.yaml Secret) ---
INFLUXDB_URL = os.environ.get("INFLUXDB_URL")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET")

# =============================================================================
# INITIALIZATION
# =============================================================================
shm = None
shared_frame_array = None

def get_shared_memory():
    """Connects to shared memory, retrying if needed."""
    global shm, shared_frame_array
    while shm is None:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=shm.buf)
            logger.info("Web app connected to shared memory for video feed.")
        except FileNotFoundError:
            logger.warning("Shared memory block not found. Is camera_manager running? Retrying in 5s...")
            time.sleep(5)

# 2. Initialize Azure Computer Vision Client
if AZURE_VISION_KEY and AZURE_VISION_ENDPOINT:
    cv_client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_KEY))
    logger.info("Computer Vision client initialized.")
else:
    logger.error("Azure credentials not found. Analysis will be disabled.")
    cv_client = None

# 3. Initialize InfluxDB Client
influx_client = None
write_api = None
query_api = None
if INFLUXDB_URL and INFLUXDB_TOKEN and INFLUXDB_ORG and INFLUXDB_BUCKET:
    try:
        influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        query_api = influx_client.query_api()
        logger.info("InfluxDB client initialized.")
    except Exception as e:
        logger.error(f"Could not initialize InfluxDB client: {e}")
else:
    logger.warning("InfluxDB credentials missing. History feature will be disabled.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def draw_bounding_boxes(frame, objects):
    """Draws bounding boxes and confidence scores onto the frame."""
    frame_with_boxes = frame.copy()
    for obj in objects:
        confidence = obj.get('confidence', 0)
        # Only draw if confidence meets the threshold
        if confidence >= MIN_OBJECT_CONFIDENCE:
            box = obj.get('rectangle', {})
            x, y, w, h = box.get('x',0), box.get('y',0), box.get('w',0), box.get('h',0)
            
            # Draw rectangle (color BGR: Blue)
            cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Put label and confidence text
            label = f"{obj.get('object_property', 'N/A')} {confidence:.0%}"
            cv2.putText(frame_with_boxes, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame_with_boxes

def save_analysis_to_db(analysis_data, frame_b64):
    """Writes the analysis result and the annotated frame to InfluxDB."""
    if not write_api:
        logger.warning("Attempted to save analysis, but InfluxDB client is not ready.")
        return

    try:
        # Prepare fields for InfluxDB point
        point = Point("analysis_record").tag("source", "rtsp-stream")

        # Add data fields
        point.field("description", analysis_data.get('description_text', ''))
        point.field("tags_json", json.dumps(analysis_data.get('tags', [])))
        point.field("objects_json", json.dumps(analysis_data.get('objects', [])))
        point.field("object_count", len(analysis_data.get('objects', [])))
        point.field("frame_b64", frame_b64) # Store the annotated frame as base64 string

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        logger.info(f"Analysis saved to InfluxDB with {len(analysis_data.get('objects', []))} objects.")
    except Exception as e:
        logger.error(f"Failed to write analysis to InfluxDB: {e}")


def run_analysis_task():
    """Captures the current frame, sends it to Azure, and returns the analysis."""
    if shared_frame_array is None:
        get_shared_memory() # Attempt to reconnect
        if shared_frame_array is None:
             return {'error': 'Shared memory not available.'}
             
    if cv_client is None:
        return {'error': 'Azure client not initialized.'}

    # 1. Capture the current frame from shared memory
    current_frame = shared_frame_array.copy()

    # 2. Encode frame to memory buffer (JPEG format)
    ret, buffer = cv2.imencode('.jpg', current_frame)
    if not ret:
        return {'error': 'Failed to encode frame.'}
    image_bytes_io = io.BytesIO(buffer.tobytes())

    # 3. Call Azure Computer Vision
    try:
        analysis_features = ['Description', 'Tags', 'Objects']
        analysis = cv_client.analyze_image_in_stream(image_bytes_io, analysis_features, language="en")
        
        # 4. Filter objects by confidence score (Server-side filtering)
        filtered_objects = [
            obj for obj in analysis.objects if obj.confidence >= MIN_OBJECT_CONFIDENCE
        ]

        # 5. Compile results into simple dicts
        serializable_objects = [
            {'object_property': obj.object_property, 'confidence': obj.confidence, 'rectangle': obj.rectangle.as_dict()}
            for obj in filtered_objects
        ]
        serializable_tags = [
            {'name': tag.name, 'confidence': tag.confidence}
            for tag in analysis.tags
        ]
        description_text = ""
        if analysis.description and analysis.description.captions:
            description_text = analysis.description.captions[0].text
        
        results = {
            'description_text': description_text,
            'tags': serializable_tags,
            'objects': serializable_objects
        }
        
        # 6. Prepare the frame for saving (with boxes drawn for history)
        frame_with_boxes = draw_bounding_boxes(current_frame, serializable_objects)
        _, buffer_boxes = cv2.imencode('.jpg', frame_with_boxes)
        frame_b64 = base64.b64encode(buffer_boxes.tobytes()).decode('utf-8')
        
        # 7. Save to InfluxDB
        save_analysis_to_db(results, frame_b64)

        return results

    except Exception as e:
        logger.error(f"Azure API call failed: {e}")
        return {'error': f"Azure Analysis Error: {str(e)}"}


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', min_confidence=MIN_OBJECT_CONFIDENCE)

def generate_frames():
    """Generator function that yields frames from shared memory."""
    if shared_frame_array is None:
        get_shared_memory() # Attempt to reconnect
    
    if shared_frame_array is None:
        logger.error("Cannot stream: Shared memory not connected.")
        # Optional: Yield a "camera unavailable" image here
        return

    while True:
        try:
            # Copy the current frame from shared memory
            frame = shared_frame_array.copy()
            
            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpeg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1/30) # 30 FPS cap

        except Exception as e:
            logger.error(f"Error streaming frame: {e}")
            # This can happen if the shared memory is closed
            get_shared_memory() # Try to reconnect
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_current_frame', methods=['POST'])
def analyze_current_frame():
    """Triggers object detection and returns results."""
    logger.info("Received request to analyze frame.")
    result = run_analysis_task()
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/get_analysis_history', methods=['GET'])
def get_analysis_history():
    """Retrieves the last N analysis records from InfluxDB."""
    if not query_api:
        return jsonify({'error': 'InfluxDB client is not ready.'}), 503

    try:
        # --- FIX for Schema Collision Error ---
        # 1. Query for all STRING fields first.
        query_strings = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -30d) 
          |> filter(fn: (r) => r._measurement == "analysis_record" and r._field != "object_count")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 50) 
        '''
        
        tables_strings = query_api.query(query_strings, org=INFLUXDB_ORG)
        
        time_to_data = {}
        for table in tables_strings:
            for record in table.records:
                ts = record.values['_time'].isoformat()
                if ts not in time_to_data:
                    time_to_data[ts] = {'timestamp': ts}
                
                field_name = record.values['_field']
                field_value = record.values['_value']
                time_to_data[ts][field_name] = field_value

        # 2. Query for the INTEGER field 'object_count' separately.
        query_int = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -30d) 
          |> filter(fn: (r) => r._measurement == "analysis_record" and r._field == "object_count")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 50) 
        '''
        tables_int = query_api.query(query_int, org=INFLUXDB_ORG)
        
        # 3. Merge integer data back into the main dictionary.
        for table in tables_int:
            for record in table.records:
                ts = record.values['_time'].isoformat()
                if ts in time_to_data: # Only add if we have the string data
                    time_to_data[ts]['object_count'] = record.values['_value']


        # Convert dict to a list
        history_list = list(time_to_data.values())
        
        # Filter out any incomplete records (e.g., if one query missed a timestamp)
        history_list = [
            data for data in history_list
            if 'description' in data and 'frame_b64' in data and 'object_count' in data
        ]
        
        # Sort by timestamp descending
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(history_list[:10]) # Return top 10 recent complete records

    except Exception as e:
        logger.error(f"Failed to query InfluxDB for history: {e}")
        return jsonify({'error': f"Failed to retrieve history: {str(e)}"}), 500

@app.route('/get_historical_frame/<timestamp>', methods=['GET'])
def get_historical_frame(timestamp):
    """Retrieves the full data record for a specific timestamp."""
    # This route is no longer needed, as /get_analysis_history now sends
    # all data needed for the modal, including the frame_b64.
    # We will leave it here but it's unused by the new index.html.
    return jsonify({'error': 'This endpoint is deprecated.'}), 404

if __name__ == '__main__':
    # This is for local development only
    app.run(host='0.0.0.0', port=8000, debug=True)
EOF

echo "--- Created app.py ---"

# --- 7. camera_manager.py ---
cat <<'EOF' > $PROJECT_DIR/camera_manager.py
import cv2
import numpy as np
import time
import logging
import os
import multiprocessing.shared_memory as shared_memory
import signal
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('camera_manager')

# --- Frame Config (Must match app.py) ---
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
FRAME_CHANNELS = 3
SHM_NAME = 'object_detector_frame_buffer'
SHARED_BUFFER_SIZE = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS

# --- Environment Variables ---
RTSP_STREAM_URL = os.environ.get("RTSP_STREAM_URL")
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "local") # default to 'local'
CONFIG_FILE_PATH = "/app/config/camera_source.txt"

# Set OpenCV options for FFMPEG to be more resilient
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'

# --- Global objects for cleanup ---
shm = None
camera = None

def cleanup(signum, frame):
    """Graceful cleanup function."""
    global shm, camera
    logger.info(f"Caught signal {signum}. Shutting down...")
    if camera and camera.isOpened():
        camera.release()
    if shm:
        shm.close()
        try:
            shm.unlink() # Unlink the shared memory on exit
        except FileNotFoundError:
            pass
    logger.info("Shutdown complete.")
    sys.exit(0)

def get_camera_source():
    """Reads the desired camera source from the config file."""
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            source = f.read().strip().lower()
            if source in ['local', 'rtsp']:
                return source
    except Exception:
        pass # File not found or invalid, fall back to env var
    return CAMERA_SOURCE

def run_camera():
    """Main camera loop with robust reconnection."""
    global shm, camera
    
    # Register signal handlers for graceful shutdown (e.g., kubectl delete pod)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        # Create the shared memory block
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHARED_BUFFER_SIZE)
        logger.info(f"Shared memory block '{SHM_NAME}' created.")
    except FileExistsError:
        # If it already exists, just connect to it
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=SHARED_BUFFER_SIZE)
        logger.info(f"Shared memory block '{SHM_NAME}' already exists, connecting.")
    
    # Create a NumPy array backed by the shared memory buffer
    shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=shm.buf)
    
    while True:
        source_type = get_camera_source()
        camera_path = ""
        
        try:
            if source_type == 'local':
                camera_path = 0 # Use /dev/video0
                camera = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
            elif source_type == 'rtsp':
                if not RTSP_STREAM_URL:
                    raise ValueError("CAMERA_SOURCE is 'rtsp' but RTSP_STREAM_URL is not set.")
                camera_path = RTSP_STREAM_URL
                camera = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)
            else:
                raise ValueError(f"Invalid CAMERA_SOURCE: {source_type}")

            if not camera or not camera.isOpened():
                raise IOError(f"Failed to open camera for source: {camera_path}")
            
            logger.info(f"Camera opened successfully ({source_type}). Starting frame capture.")

            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame. Breaking to reconnect...")
                    break 
                
                # Resize frame if it doesn't match the shared memory dimensions
                if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                
                # Write the frame directly into the shared memory buffer
                shared_frame_array[:] = frame[:]
                
                # Sleep to cap capture rate (e.g., 30fps)
                time.sleep(1/30) 
        
        except Exception as e:
            logger.error(f"Error in main camera loop: {e}")
        
        finally:
            if camera:
                camera.release()
            logger.info("Camera resource released. Waiting 5s before reconnect...")
            time.sleep(5)

if __name__ == '__main__':
    run_camera()
EOF

echo "--- Created camera_manager.py ---"

# --- 8. templates/index.html ---
cat <<'EOF' > $PROJECT_DIR/templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTSP Object Detector</title>
    <!-- 
      Tailwind is used for rapid prototyping. 
      For production, you should use the Tailwind CLI/PostCSS.
    -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .video-container {
            width: 100%;
            max-width: 1280px;
            aspect-ratio: 16 / 9;
            position: relative;
            background-color: #000;
        }
        #videoFeed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        #overlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        /* Modal styles */
        .modal {
            transition: opacity 0.25s ease;
        }
        .modal-content {
            transition: all 0.25s ease;
            max-height: 90vh;
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-900">

    <div class="container mx-auto p-4 max-w-7xl">
        <h1 class="text-3xl font-bold text-center text-blue-700 mb-6">RTSP Object Detector</h1>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            <!-- Main Content: Video and Analysis -->
            <div class="lg:col-span-2 bg-white p-6 rounded-lg shadow-lg">
                <div class="video-container rounded-lg overflow-hidden shadow-inner border border-gray-300">
                    <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Stream">
                    <canvas id="overlayCanvas"></canvas>
                </div>

                <!-- Controls -->
                <div class="mt-6 flex flex-col sm:flex-row gap-4">
                    <button id="analyzeButton" class="flex-1 bg-green-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:bg-green-700 transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed">
                        <span id="analyzeSpinner" class="hidden mr-2 w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                        <span id="analyzeText">Analyze Current Frame</span>
                    </button>
                    <button id="localCamButton" class="flex-1 bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:bg-blue-700 transition duration-300">
                        Use Local Camera
                    </button>
                    <button id="rtspCamButton" class="flex-1 bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:bg-blue-700 transition duration-300">
                        Use RTSP Stream
                    </button>
                </div>

                <!-- Status Message -->
                <div id="statusMessage" class="mt-4 text-center font-medium min-h-[1.5em]"></div>

                <!-- Current Analysis Results -->
                <div class="mt-6">
                    <h2 class="text-2xl font-semibold border-b border-gray-300 pb-2 mb-4">Current Analysis</h2>
                    <div id="currentAnalysisResults" class="space-y-3">
                        <p><strong>Description:</strong> <span id="descriptionText" class="text-gray-700">No analysis yet.</span></p>
                        <p><strong>Tags:</strong> <span id="tagsList" class="text-gray-700">No analysis yet.</span></p>
                        <div>
                            <p><strong>Detected Objects (Min. {{ (min_confidence * 100) }}% confidence):</strong></p>
                            <ul id="objectsList" class="list-disc list-inside text-gray-700 pl-4">
                                <li>No analysis yet.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Sidebar: Analysis History -->
            <div class="lg:col-span-1 bg-white p-6 rounded-lg shadow-lg">
                <h2 class="text-2xl font-semibold border-b border-gray-300 pb-2 mb-4">Analysis History</h2>
                <button id="refreshHistoryBtn" class="w-full bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-lg transition duration-300 mb-4">
                    Refresh History
                </button>
                <div id="historyError" class="text-red-600 mb-4 hidden"></div>
                <div id="historyList" class="space-y-3 max-h-[600px] overflow-y-auto">
                    <p id="historyLoading" class="text-gray-500">Loading history...</p>
                    <!-- History items will be injected here -->
                </div>
            </div>

        </div>
    </div>

    <!-- History Detail Modal -->
    <div id="historyModal" class="modal fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-75 opacity-0 pointer-events-none">
        <div class="modal-content bg-white rounded-lg shadow-2xl w-full max-w-4xl transform scale-95 opacity-0 overflow-y-auto">
            <div class="flex justify-between items-center p-6 border-b">
                <h2 class="text-2xl font-semibold">Historical Analysis</h2>
                <button id="closeModalBtn" class="text-gray-500 hover:text-gray-800 text-3xl">&times;</button>
            </div>
            <div class="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Modal Image -->
                <div>
                    <h3 class="text-xl font-semibold mb-2">Analyzed Frame</h3>
                    <img id="modalImage" src="" alt="Analyzed Frame" class="rounded-lg border w-full">
                </div>
                <!-- Modal Details -->
                <div class="space-y-4">
                    <p><strong>Timestamp:</strong> <span id="modalTimestamp" class="text-gray-700"></span></p>
                    <p><strong>Description:</strong> <span id="modalDescription" class="text-gray-700"></span></p>
                    <div>
                        <p><strong>Tags:</strong></p>
                        <div id="modalTags" class="text-gray-700 flex flex-wrap gap-2 mt-2">
                            <!-- Tags will be injected here -->
                        </div>
                    </div>
                    <div>
                        <p><strong>Detected Objects:</strong></p>
                        <ul id="modalObjects" class="list-disc list-inside text-gray-700 pl-4">
                            <!-- Objects will be injected here -->
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // --- DOM Elements ---
        const videoFeed = document.getElementById('videoFeed');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const ctx = overlayCanvas.getContext('2d');
        
        const analyzeButton = document.getElementById('analyzeButton');
        const analyzeSpinner = document.getElementById('analyzeSpinner');
        const analyzeText = document.getElementById('analyzeText');
        
        const localCamButton = document.getElementById('localCamButton');
        const rtspCamButton = document.getElementById('rtspCamButton');
        const statusMessage = document.getElementById('statusMessage');

        // Current Analysis Elements
        const descriptionText = document.getElementById('descriptionText');
        const tagsList = document.getElementById('tagsList');
        const objectsList = document.getElementById('objectsList');

        // History Elements
        const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
        const historyList = document.getElementById('historyList');
        const historyLoading = document.getElementById('historyLoading');
        const historyError = document.getElementById('historyError');
        let analysisHistoryCache = []; // Cache to hold full history data

        // Modal Elements
        const historyModal = document.getElementById('historyModal');
        const closeModalBtn = document.getElementById('closeModalBtn');
        const modalImage = document.getElementById('modalImage');
        const modalTimestamp = document.getElementById('modalTimestamp');
        const modalDescription = document.getElementById('modalDescription');
        const modalTags = document.getElementById('modalTags');
        const modalObjects = document.getElementById('modalObjects');

        // --- Event Listeners ---
        
        // Analyze Button
        analyzeButton.addEventListener('click', handleAnalyzeFrame);
        
        // Camera Switch Buttons
        localCamButton.addEventListener('click', () => switchCameraSource('local'));
        rtspCamButton.addEventListener('click', () => switchCameraSource('rtsp'));

        // History Buttons
        refreshHistoryBtn.addEventListener('click', loadHistory);

        // Modal Close Buttons
        closeModalBtn.addEventListener('click', closeModal);
        historyModal.addEventListener('click', (e) => {
            if (e.target === historyModal) closeModal();
        });

        // --- Initialization ---
        window.addEventListener('load', () => {
            resizeCanvas();
            loadHistory();
        });
        window.addEventListener('resize', resizeCanvas);

        // --- Functions ---

        function resizeCanvas() {
            // Make canvas overlay match the video feed's rendered size
            overlayCanvas.width = videoFeed.clientWidth;
            overlayCanvas.height = videoFeed.clientHeight;
        }

        function setAnalyzeButtonState(isLoading, message = 'Analyze Current Frame') {
            analyzeButton.disabled = isLoading;
            analyzeText.textContent = message;
            if (isLoading) {
                analyzeSpinner.classList.remove('hidden');
            } else {
                analyzeSpinner.classList.add('hidden');
            }
        }

        function showStatus(message, isError = false) {
            statusMessage.textContent = message;
            statusMessage.style.color = isError ? '#EF4444' : '#10B981';
            if (!isError) {
                setTimeout(() => statusMessage.textContent = '', 3000);
            }
        }

        async function switchCameraSource(source) {
            showStatus(`Switching to ${source} stream...`);
            localCamButton.disabled = true;
            rtspCamButton.disabled = true;

            try {
                // This is a "fire-and-forget" request to the config file
                // The camera_manager.py will pick up the change
                await fetch(`/switch_camera?source=${source}`, { method: 'POST' });
                showStatus(`Request sent to switch to ${source}. Stream will update shortly.`);
                videoFeed.src = `/video_feed?t=${new Date().getTime()}`; // Force reload
            } catch (error) {
                console.error('Error switching camera:', error);
                showStatus('Error: Could not send switch request.', true);
            } finally {
                localCamButton.disabled = false;
                rtspCamButton.disabled = false;
            }
        }

        async function handleAnalyzeFrame() {
            setAnalyzeButtonState(true, 'Analyzing...');
            showStatus('');
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            clearCurrentAnalysisDetails();

            try {
                const response = await fetch('/analyze_current_frame', { method: 'POST' });
                
                if (!response.ok) {
                    // Try to parse error from server
                    let errorMsg = `HTTP error! Status: ${response.status}`;
                    try {
                        const errData = await response.json();
                        errorMsg = errData.error || errorMsg;
                    } catch (e) {
                        // Response wasn't JSON, maybe HTML error page
                         const textError = await response.text();
                         throw new Error(`Analysis API call failed: ${response.status}. Response: ${textError.substring(0, 100)}...`);
                    }
                    throw new Error(errorMsg);
                }
                
                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                showStatus('Analysis complete!', false);
                populateCurrentAnalysisDetails(data);
                drawBoxesOnCanvas(data.objects);
                loadHistory(); // Refresh history after analysis

            } catch (error) {
                console.error('Analysis error:', error);
                let errorText = error.message;
                // Handle JSON parse errors from 404/405s
                if (error instanceof SyntaxError && error.message.includes("is not valid JSON")) {
                    errorText = `Analysis API call failed. Is the server running the correct code? (Received non-JSON response)`;
                }
                showStatus(`Error: ${errorText}`, true);
            } finally {
                setAnalyzeButtonState(false);
            }
        }

        function populateCurrentAnalysisDetails(data) {
            descriptionText.textContent = data.description_text || 'No description available.';
            tagsList.textContent = data.tags?.map(t => `${t.name} (${(t.confidence * 100).toFixed(0)}%)`).join(', ') || 'No tags available.';
            
            if (data.objects && data.objects.length > 0) {
                objectsList.innerHTML = '';
                data.objects.forEach(obj => {
                    const li = document.createElement('li');
                    li.textContent = `${obj.object_property} (${(obj.confidence * 100).toFixed(0)}%)`;
                    objectsList.appendChild(li);
                });
            } else {
                objectsList.innerHTML = '<li>No objects detected.</li>';
            }
        }
        
        function clearCurrentAnalysisDetails() {
            descriptionText.textContent = '...';
            tagsList.textContent = '...';
            objectsList.innerHTML = '<li>...</li>';
        }

        function drawBoxesOnCanvas(objects) {
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            
            // Get scaling factors
            const scaleX = overlayCanvas.width / 1280; // 1280 is the original frame width
            const scaleY = overlayCanvas.height / 720; // 720 is the original frame height

            ctx.strokeStyle = '#00BFFF'; // Deep Sky Blue
            ctx.lineWidth = 2;
            ctx.font = '16px Arial';
            ctx.fillStyle = '#00BFFF';

            objects.forEach(obj => {
                const box = obj.rectangle;
                const x = box.x * scaleX;
                const y = box.y * scaleY;
                const w = box.w * scaleX;
                const h = box.h * scaleY;
                
                ctx.strokeRect(x, y, w, h);
                
                const label = `${obj.object_property} (${(obj.confidence * 100).toFixed(0)}%)`;
                ctx.fillText(label, x, y > 20 ? y - 5 : y + h + 15);
            });
        }

        // --- History Functions ---

        async function loadHistory() {
            historyLoading.classList.remove('hidden');
            historyError.classList.add('hidden');
            historyList.innerHTML = ''; // Clear old list
            historyList.appendChild(historyLoading);

            try {
                const response = await fetch('/get_analysis_history');
                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || `HTTP Error ${response.status}`);
                }
                
                const data = await response.json();
                analysisHistoryCache = data; // Store full data in cache
                renderHistoryList(data);

            } catch (error) {
                console.error('Error loading history:', error);
                 let errorText = error.message;
                if (error instanceof SyntaxError && error.message.includes("is not valid JSON")) {
                    errorText = `History API call failed. Is the server running? (Received non-JSON response)`;
                }
                historyError.textContent = `Error: ${errorText}`;
                historyError.classList.remove('hidden');
            } finally {
                historyLoading.classList.add('hidden');
            }
        }

        function renderHistoryList(historyData) {
            historyList.innerHTML = ''; // Clear "Loading..."
            if (historyData.length === 0) {
                historyList.innerHTML = '<p class="text-gray-500">No history found.</p>';
                return;
            }

            historyData.forEach((item, index) => {
                const date = new Date(item.timestamp);
                const li = document.createElement('li');
                li.className = 'p-3 rounded-lg border hover:bg-gray-50 cursor-pointer flex justify-between items-center';
                li.innerHTML = `
                    <div>
                        <span class="font-semibold">${date.toLocaleString()}</span>
                        <span class="block text-sm text-gray-600">${item.description || 'No description'}</span>
                    </div>
                    <span class="text-sm font-medium text-blue-600">${item.object_count} Object(s)</span>
                `;
                li.addEventListener('click', () => openModal(index));
                historyList.appendChild(li);
            });
        }

        function openModal(index) {
            const item = analysisHistoryCache[index];
            if (!item) return;
            
            modalImage.src = `data:image/jpeg;base64,${item.frame_b64}`;
            modalTimestamp.textContent = new Date(item.timestamp).toLocaleString();
            modalDescription.textContent = item.description || 'N/A';
            
            // Populate Tags
            modalTags.innerHTML = '';
            const tags = JSON.parse(item.tags_json || '[]');
            if (tags.length > 0) {
                tags.forEach(tag => {
                    const tagEl = document.createElement('span');
                    tagEl.className = 'bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded';
                    tagEl.textContent = `${tag.name} (${(tag.confidence * 100).toFixed(0)}%)`;
                    modalTags.appendChild(tagEl);
                });
            } else {
                modalTags.innerHTML = '<span class="text-gray-500">No tags.</span>';
            }
            
            // Populate Objects
            modalObjects.innerHTML = '';
            const objects = JSON.parse(item.objects_json || '[]');
            if (objects.length > 0) {
                objects.forEach(obj => {
                    const li = document.createElement('li');
                    li.textContent = `${obj.object_property} (${(obj.confidence * 100).toFixed(0)}%)`;
                    modalObjects.appendChild(li);
                });
            } else {
                modalObjects.innerHTML = '<li>No objects detected.</li>';
            }

            // Show modal
            historyModal.classList.remove('opacity-0', 'pointer-events-none');
            historyModal.querySelector('.modal-content').classList.remove('scale-95', 'opacity-0');
        }

        function closeModal() {
            historyModal.classList.add('opacity-0', 'pointer-events-none');
            historyModal.querySelector('.modal-content').classList.add('scale-95', 'opacity-0');
        }

    </script>
</body>
</html>
EOF

echo "--- Created templates/index.html ---"

echo "---"
echo "✅ All files generated in $PROJECT_DIR/"
echo "---"
echo "IMPORTANT: Remember to manually add your 'camera_unavailable.jpg' file to the '$PROJECT_DIR/static/' directory."
echo "Next, follow the instructions in '$PROJECT_DIR/README.md' to build, push, and deploy."


