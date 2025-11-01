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
