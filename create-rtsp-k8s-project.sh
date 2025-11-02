#!/bin/bash
# This script creates the complete directory structure and all files
# for the Kubernetes RTSP object detector project.

set -e

PROJECT_DIR="rtsp-k8s-project"

echo "Creating project directory '$PROJECT_DIR'..."
mkdir -p $PROJECT_DIR/templates
mkdir -p $PROJECT_DIR/static
cd $PROJECT_DIR

# --- 1. Dockerfile ---
echo "Writing Dockerfile..."
cat <<'EOF' > Dockerfile
# Use a Python slim base image which is Debian-based, similar to the script's environment
FROM python:3.9-slim-bullseye

# Install ffmpeg and other OS dependencies required by opencv-python and your app
# libgstreamer1.0-0 and plugins-base are often needed for RTSP streams in OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set up the application directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and static files
COPY app.py .
COPY camera_manager.py .
COPY templates/index.html ./templates/index.html
COPY static/camera_unavailable.jpg ./static/camera_unavailable.jpg

# Note: The web app will run on port 8000
EXPOSE 8000

# The default command is handled in the Kubernetes Deployment manifest
# This image will be used by *both* containers in the Pod
EOF

# --- 2. k8s-manifest.yaml ---
echo "Writing k8s-manifest.yaml..."
cat <<'EOF' > k8s-manifest.yaml
# --- 1. Secret ---
# Stores your sensitive credentials
apiVersion: v1
kind: Secret
metadata:
  name: rtsp-app-secret
  namespace: default
type: Opaque
stringData:
  # !! IMPORTANT: Update these values with your real credentials
  AZURE_VISION_KEY: "YOUR_AZURE_KEY_HERE"
  AZURE_VISION_ENDPOINT: "YOUR_AZURE_ENDPOINT_HERE"
  RTSP_URL: "rtsp://192.168.1.250/axis-media/media.amp"

---
# --- 2. Deployment ---
# Deploys a single Pod with two containers that share memory
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtsp-object-detector
  namespace: default
  labels:
    app: rtsp-object-detector
spec:
  # This application is stateful (due to shared memory) and cannot be scaled horizontally
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
        # Create a shared in-memory volume for the two containers
        - name: frame-buffer
          emptyDir:
            medium: Memory
            # Set a size limit to prevent memory exhaustion
            sizeLimit: "256Mi"

      containers:
        # --- Container 1: The Web Application (app.py) ---
        - name: web-app
          # !! IMPORTANT: Update this with your container registry and image name
          image: "your-registry/rtsp-object-detector:latest"
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
          command: ["gunicorn"]
          args: [
            "--workers", "3",
            "--worker-class", "gevent",
            "--bind", "0.0.0.0:8000",
            "app:app"
          ]
          env:
            # Load credentials from the Secret
            - name: AZURE_VISION_SUBSCRIPTION_KEY
              valueFrom:
                secretKeyRef:
                  name: rtsp-app-secret
                  key: AZURE_VISION_KEY
            - name: AZURE_VISION_ENDPOINT
              valueFrom:
                secretKeyRef:
                  name: rtsp-app-secret
                  key: AZURE_VISION_ENDPOINT
          volumeMounts:
            # Mount the shared memory volume
            - name: frame-buffer
              mountPath: /dev/shm
          readinessProbe:
            httpGet:
              path: /
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10

        # --- Container 2: The Camera Manager (camera_manager.py) ---
        - name: camera-manager
          # !! IMPORTANT: Use the *same image* as the web-app
          image: "your-registry/rtsp-object-detector:latest"
          imagePullPolicy: Always
          command: ["python3"]
          args: ["camera_manager.py"]
          env:
            # Load the RTSP URL from the Secret
            - name: RTSP_STREAM_URL
              valueFrom:
                secretKeyRef:
                  name: rtsp-app-secret
                  key: RTSP_URL
            # Set the camera source directly.
            # The 'local' camera option is not viable in Kubernetes without
            # privileged access, so we hardcode this to 'rtsp'.
            - name: CAMERA_SOURCE
              value: "rtsp"
          volumeMounts:
            # Mount the *same* shared memory volume
            - name: frame-buffer
              mountPath: /dev/shm

---
# --- 3. Service ---
# Exposes the web-app container internally in the cluster
apiVersion: v1
kind: Service
metadata:
  name: rtsp-object-detector-svc
  namespace: default
spec:
  type: ClusterIP
  selector:
    # This must match the Deployment's labels
    app: rtsp-object-detector
  ports:
    - port: 80
      # This must match the web-app containerPort
      targetPort: 8000

---
# --- 4. Ingress ---
# Exposes the Service to the outside world via your nginx-ingress-controller
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rtsp-object-detector-ingress
  namespace: default
  annotations:
    # This assumes your RKE2 cluster's default IngressClass is 'nginx'
    kubernetes.io/ingress.class: "nginx"
spec:
  rules:
    # !! IMPORTANT: Update this host to your desired domain
    - host: "rtsp-detector.your-domain.com"
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                # This must match the Service name
                name: rtsp-object-detector-svc
                port:
                  # This must match the Service port
                  number: 80
EOF

# --- 3. app.py ---
echo "Writing app.py..."
cat <<'EOF' > app.py
import gevent.monkey
gevent.monkey.patch_all()

import os
import cv2
import numpy as np
import time
import logging
import io
import queue
from flask import Flask, render_template, Response, jsonify
import multiprocessing.shared_memory as shared_memory
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)

# --- App Config ---
# The /tmp/camera_source.txt is removed; configuration is now via Env Var
SHM_NAME = 'object_detector_frame_buffer' # Must match camera_manager.py

# --- Frame Config ---
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
FRAME_CHANNELS = 3

# --- Azure Config ---
COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ.get("AZURE_VISION_SUBSCRIPTION_KEY")
COMPUTER_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT")

# =============================================================================
# INITIALIZATION
# =============================================================================
computervision_client = None
analysis_executor = ThreadPoolExecutor(max_workers=1)
camera_unavailable_image_bytes = None

try:
    with open("static/camera_unavailable.jpg", "rb") as f:
        camera_unavailable_image_bytes = f.read()
except Exception as e:
    logger.error(f"Could not load placeholder image: {e}")

if COMPUTER_VISION_SUBSCRIPTION_KEY and COMPUTER_VISION_ENDPOINT:
    try:
        computervision_client = ComputerVisionClient(
            endpoint=COMPUTER_VISION_ENDPOINT,
            credentials=CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY)
        )
        logger.info("Computer Vision client initialized.")
    except Exception as e:
        logger.error(f"Could not initialize Azure client: {e}")
else:
    logger.warning("Azure credentials not found. Analysis will be disabled.")

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        existing_shm = None
        while True:
            try:
                if existing_shm is None:
                    # Connect to the shared memory block created by the camera_manager container
                    existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
                    logger.info("Web app connected to shared memory for video feed.")

                shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=existing_shm.buf)
                frame = shared_frame_array.copy()

                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except FileNotFoundError:
                logger.warning("Shared memory file not found. Camera manager might be restarting.")
                if existing_shm:
                    existing_shm.close()
                    existing_shm = None
                if camera_unavailable_image_bytes:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + camera_unavailable_image_bytes + b'\r\n')
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in video feed generator: {e}")
                time.sleep(1)

            # Frame rate control
            time.sleep(0.033)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- REMOVED /switch_camera ---
# This endpoint is removed as it relied on 'systemctl' and 'sudo',
# which are not compatible with a Kubernetes environment.
# Camera source is now configured via the 'CAMERA_SOURCE' env var
# in the camera-manager container at deployment time.

@app.route('/analyze_current_frame')
def analyze_current_frame():
    if not computervision_client:
        return jsonify({"status": "error", "message": "Azure analysis client is not configured."}), 500

    shm = None
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=shm.buf)
        frame_to_analyze = shared_frame_array.copy()
        shm.close()

        if not frame_to_analyze.any():
            return jsonify({"status": "error", "message": "Received an empty frame from camera."}), 400

        ret, jpeg_bytes = cv2.imencode('.jpg', frame_to_analyze)
        if not ret:
            return jsonify({"status": "error", "message": "Failed to encode frame for analysis."}), 500

        def run_analysis_task(frame_b, result_queue):
            try:
                analysis = computervision_client.analyze_image_in_stream(io.BytesIO(frame_b), ["Objects", "Tags", "Description"])
                data = {"description": analysis.description.as_dict() if analysis.description else {}, "tags": [t.as_dict() for t in analysis.tags] if analysis.tags else [], "objects": [], "hat_objects": []}
                if analysis.objects:
                    for obj in analysis.objects:
                        obj_dict = obj.as_dict()
                        data["objects"].append(obj_dict)
                        if obj.object_property and obj.object_property.lower() in ['hat', 'cap']:
                            data["hat_objects"].append(obj_dict)
                result_queue.put({"status": "completed", "analysis_data": data})
            except Exception as ex:
                logger.error(f"Azure analysis failed: {ex}")
                result_queue.put({"status": "failed", "message": str(ex)})

        result_queue = queue.Queue()
        analysis_executor.submit(run_analysis_task, jpeg_bytes.tobytes(), result_queue)
        result = result_queue.get(timeout=20)

        if result['status'] == 'completed':
            return jsonify({"status": "success", "analysis_data": result['analysis_data']})
        else:
            return jsonify({"status": "error", "message": result['message']}), 500

    except FileNotFoundError:
        return jsonify({"status": "error", "message": "Camera feed not active (shared memory not found)."}), 404
    except queue.Empty:
        return jsonify({"status": "error", "message": "Analysis processing timed out."}), 504
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred during analysis."}), 500
    finally:
        if shm:
            shm.close()

# This is needed if running directly with 'python app.py' for testing
# But gunicorn will be the entrypoint in production
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOF

# --- 4. camera_manager.py ---
echo "Writing camera_manager.py..."
cat <<'EOF' > camera_manager.py
import gevent.monkey
gevent.monkey.patch_all()

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RTSP_STREAM_URL = os.environ.get("RTSP_STREAM_URL")
# Get camera source from environment variable
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "rtsp").lower()

# Set a long timeout (60 seconds) for FFMPEG to be more patient with streams
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;60000000'

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
FRAME_CHANNELS = 3
SHARED_BUFFER_SIZE = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS
SHM_NAME = 'object_detector_frame_buffer'

# --- Global objects for cleanup ---
shm = None
camera = None

def cleanup(signum, frame):
    """Graceful cleanup function."""
    global shm, camera
    logging.info(f"Caught signal {signum}. Shutting down camera_manager...")
    if camera and camera.isOpened():
        camera.release()
        logging.info("Camera stream released.")
    if shm:
        shm.close()
        try:
            # Unlink the shared memory block on shutdown
            shm.unlink()
            logging.info("Shared memory unlinked.")
        except FileNotFoundError:
            pass
    logging.info("Camera manager shutdown complete.")
    sys.exit(0)

# --- REMOVED get_camera_source() ---
# This function is no longer needed as the source is read
# directly from the CAMERA_SOURCE environment variable.

def run_camera():
    """Main camera loop with robust reconnection."""
    global shm, camera
    
    # Register signal handlers for graceful shutdown (e.g., from 'kubectl delete pod')
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        # Create the shared memory block.
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHARED_BUFFER_SIZE)
        logging.info(f"Shared memory block '{SHM_NAME}' created.")
    except FileExistsError:
        # If it already exists (e.g., from a previous crash), unlink and recreate
        logging.warning("Shared memory block already exists. Unlinking and recreating.")
        try:
            temp_shm = shared_memory.SharedMemory(name=SHM_NAME)
            temp_shm.close()
            temp_shm.unlink()
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHARED_BUFFER_SIZE)

    shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=shm.buf)
    
    while True:
        try:
            source_type = CAMERA_SOURCE
            source_path = RTSP_STREAM_URL if source_type == 'rtsp' else 0
            
            if source_type == 'rtsp' and not RTSP_STREAM_URL:
                raise ValueError("CAMERA_SOURCE is 'rtsp' but RTSP_STREAM_URL is not set.")
                
            logging.info(f"Attempting to connect to camera source: {source_type} (Path: {source_path})")
            
            camera = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG if source_type == 'rtsp' else cv2.CAP_V4L2)
            
            if not camera or not camera.isOpened():
                raise IOError(f"Failed to open camera for source {source_type}.")
            
            logging.info("Camera opened successfully. Starting frame capture.")

            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    logging.warning("Failed to grab frame. Breaking to reconnect...")
                    break 
                
                # Resize frame if it doesn't match the shared buffer dimensions
                if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Write the frame to shared memory
                shared_frame_array[:] = frame[:]
                
                # Sleep to yield control, ~30fps
                time.sleep(0.03)
        
        except Exception as e:
            logging.error(f"Error in main camera loop: {e}")
        
        finally:
            if camera:
                camera.release()
            logging.info("Camera resource released. Waiting 5s before reconnect...")
            time.sleep(5)

if __name__ == '__main__':
    run_camera()
EOF

# --- 5. requirements.txt ---
echo "Writing requirements.txt..."
cat <<'EOF' > requirements.txt
Flask
gunicorn
opencv-python
numpy
azure-cognitiveservices-vision-computervision
msrest
tenacity
sdnotify
gevent
EOF

# --- 6. templates/index.html ---
echo "Writing templates/index.html..."
cat <<'EOF' > templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detector</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f0f0f0; color: #333; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; padding: 20px; box-sizing: border-box; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center; margin-bottom: 20px; width: 100%; max-width: 1280px; }
        h1 { color: #0056b3; }
        .video-container { position: relative; width: 100%; /* Make width responsive */ padding-top: 56.25%; /* 16:9 Aspect Ratio (720 / 1280) */ border: 2px solid #ccc; margin-bottom: 10px; overflow: hidden; background-color: #000; }
        #videoFeed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block; }
        /* Canvas removed as drawing boxes is not implemented in the script */
        /* .button-group { margin-bottom: 15px; } */
        button { padding: 10px 20px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s ease; margin: 0 5px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
        #analyzeButton { background-color: #28a745; }
        #analyzeButton:hover { background-color: #218838; }
        #statusMessage { margin-top: 15px; font-size: 1.1em; color: #555; min-height: 25px; font-weight: bold; }
        #analysisDetails { background-color: #e9ecef; padding: 15px; border-radius: 8px; text-align: left; width: 100%; max-width: 1280px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
        #analysisDetails h2 { color: #0056b3; margin-top: 0; border-bottom: 1px solid #cce5ff; padding-bottom: 5px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live Object Detector</h1>
        <div class="video-container">
            <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        <!-- 
          Camera switch buttons have been removed.
          The camera source is now set in the 'k8s-manifest.yaml' file.
          To change the source, you would edit the Deployment and re-apply it.
        -->
        <button id="analyzeButton">Analyze Frame</button>
        <div id="statusMessage">Streaming from configured RTSP source.</div>
    </div>

    <div id="analysisDetails">
        <h2>Analysis Results</h2>
        <p><strong>Description:</strong> <span id="descriptionText">No analysis yet.</span></p>
        <p><strong>Tags:</strong> <span id="tagsList">No analysis yet.</span></p>
        <p><strong>All Detected Objects:</strong></p>
        <ul id="objectsList"><li>No analysis yet.</li></ul>
    </div>

    <script>
        const videoFeed = document.getElementById('videoFeed');
        const analyzeButton = document.getElementById('analyzeButton');
        const statusMessageDiv = document.getElementById('statusMessage');

        const descriptionText = document.getElementById('descriptionText');
        const tagsList = document.getElementById('tagsList');
        const objectsList = document.getElementById('objectsList');

        // Removed all JavaScript related to 'localCamButton' and 'rtspCamButton'
        // as they no longer exist.

        analyzeButton.addEventListener('click', async () => {
            statusMessageDiv.textContent = 'Analysis in progress...';
            analyzeButton.disabled = true;
            clearAnalysisDetails();

            try {
                const response = await fetch('/analyze_current_frame');
                const data = await response.json();
                if (data.status === 'success') {
                    statusMessageDiv.textContent = `Analysis complete.`;
                    populateAnalysisDetails(data.analysis_data);
                } else {
                    statusMessageDiv.textContent = `Error: ${data.message || 'Unknown error'}`;
                }
            } catch (error) {
                console.error('Error during analysis:', error);
                statusMessageDiv.textContent = 'Error: Could not connect to analysis service.';
            } finally {
                analyzeButton.disabled = false;
            }
        });

        function populateAnalysisDetails(data) {
             descriptionText.textContent = data.description?.captions?.[0]?.text || 'No description available.';
             tagsList.textContent = data.tags?.map(t => t.name).join(', ') || 'No tags available.';
             if (data.objects && data.objects.length > 0) {
                objectsList.innerHTML = '';
                data.objects.forEach(obj => {
                    const li = document.createElement('li');
                    li.textContent = `${obj.object_property} (${Math.round(obj.confidence * 100)}%)`;
                    objectsList.appendChild(li);
                });
             } else {
                objectsList.innerHTML = '<li>No objects detected.</li>';
             }
        }
        
        function clearAnalysisDetails() {
            descriptionText.textContent = 'No analysis yet.';
            tagsList.textContent = 'No analysis yet.';
            objectsList.innerHTML = '<li>No analysis yet.</li>';
        }
    </script>
</body>
</html>
EOF

# --- 7. README.md ---
echo "Writing README.md..."
cat <<'EOF' > README.md
# RTSP Object Detector (Kubernetes)

This project runs the RTSP Object Detector application on Kubernetes.

It consists of a single Pod with two containers:
1.  **web-app**: Runs the Flask/Gunicorn server to handle web requests and Azure analysis.
2.  **camera-manager**: Runs the OpenCV script to connect to the RTSP stream and write frames to a shared memory volume.

## Deployment Steps

1.  **Add Placeholder Image:**
    * Place your `camera_unavailable.jpg` file inside the `static/` directory.

2.  **Build and Push the Docker Image:**
    * Replace `your-registry` with your Docker Hub username or private registry URL.
    * `docker build -t your-registry/rtsp-object-detector:latest .`
    * `docker push your-registry/rtsp-object-detector:latest`

3.  **Configure the Kubernetes Manifest:**
    * Open `k8s-manifest.yaml`.
    * **Crucial:** Update the placeholder values in the `Secret` (lines 10-12) with your actual Azure keys and RTSP stream URL.
    * **Crucial:** Update the `image:` name in the `Deployment` (lines 40 and 70) to match the image you just pushed.
    * **Crucial:** Update the `host:` in the `Ingress` (line 120) to the domain you want to use.

4.  **Deploy to RKE2:**
    * Make sure your `kubectl` is configured to point to your RKE2 cluster.
    * `kubectl apply -f k8s-manifest.yaml`

5.  **Access Your Application:**
    * Check the pod status: `kubectl get pods -w`
    * Once running, access the application at the host you configured in the `Ingress` (e.g., `http://rtsp-detector.your-domain.com`).
EOF

echo ""
echo "------------------------------------------------------------------"
echo "✅ Project files created in '$PROJECT_DIR' directory."
echo ""
echo "⚠️  ACTION REQUIRED:"
echo "   1. cd $PROJECT_DIR"
echo "   2. Manually add your 'camera_unavailable.jpg' file to the 'static/' directory."
echo ""
echo "After that, follow the steps in README.md to build and deploy."
echo "------------------------------------------------------------------"



