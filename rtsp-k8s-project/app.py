import gevent.monkey
gevent.monkey.patch_all()

import os
import cv2
import numpy as np
import time
import logging
import io
import subprocess
import queue
import base64
import json
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request
import multiprocessing.shared_memory as shared_memory
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from concurrent.futures import ThreadPoolExecutor

# New InfluxDB Imports
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)

# --- App Config ---
CONFIG_FILE_PATH = "/tmp/camera_source.txt"
SHM_NAME = 'object_detector_frame_buffer' # Must match camera_manager.py

# --- Frame Config ---
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
FRAME_CHANNELS = 3

# --- Analysis Config ---
MIN_OBJECT_CONFIDENCE = 0.60  # Minimum confidence (0.0 to 1.0)

# --- Azure Config ---
COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ.get("AZURE_VISION_SUBSCRIPTION_KEY")
COMPUTER_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT")

# --- InfluxDB Config (From K8s Secret) ---
INFLUXDB_URL = os.environ.get("INFLUXDB_URL")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET")

# =============================================================================
# INITIALIZATION
# =============================================================================
computervision_client = None
analysis_executor = ThreadPoolExecutor(max_workers=1)
camera_unavailable_image_bytes = None

# Azure Client Init
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

# InfluxDB Client Init
influx_client = None
write_api = None
query_api = None
if INFLUXDB_URL and INFLUXDB_TOKEN:
    try:
        influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        query_api = influx_client.query_api()
        logger.info("InfluxDB client initialized.")
    except Exception as e:
        logger.error(f"Could not initialize InfluxDB client: {e}")

# Placeholder Image Load
try:
    with open("static/camera_unavailable.jpg", "rb") as f:
        camera_unavailable_image_bytes = f.read()
except Exception as e:
    logger.error(f"Could not load placeholder image: {e}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def draw_bounding_boxes(frame, objects):
    """Draws bounding boxes on the frame for detected objects."""
    if not objects:
        return frame
        
    for obj in objects:
        # Azure Object format: { 'rectangle': {'x': x, 'y': y, 'w': w, 'h': h}, 'object_property': 'name', 'confidence': 0.XX}
        r = obj.get('rectangle', {})
        name = obj.get('object_property', 'Object')
        conf = obj.get('confidence', 0.0)
        
        # Convert coordinates to integer
        x, y, w, h = r.get('x', 0), r.get('y', 0), r.get('w', 0), r.get('h', 0)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"{name}: {conf:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    return frame

def save_analysis_to_influx(frame_b64, analysis_data):
    """Saves the analysis data and base64 image to InfluxDB."""
    if not write_api:
        logger.warning("InfluxDB write API not available. Skipping save.")
        return

    try:
        # Extract primary data for fields
        description = analysis_data['description']['captions'][0]['text'] if analysis_data['description']['captions'] else ""
        num_objects = len(analysis_data['objects'])
        
        # Create a single point for the time-series database
        point = Point("analysis_record") \
            .tag("status", "success") \
            .field("description", description) \
            .field("object_count", num_objects) \
            .field("frame_b64", frame_b64) \
            .field("tags_json", json.dumps(analysis_data['tags'])) \
            .field("objects_json", json.dumps(analysis_data['objects'])) \
            .time(datetime.utcnow(), WritePrecision.MS)
            
        write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        logger.info(f"Analysis saved to InfluxDB. Objects detected: {num_objects}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to InfluxDB: {e}")
        return False

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # ... (video_feed function remains the same) ...
    def generate_frames():
        existing_shm = None
        while True:
            try:
                if existing_shm is None:
                    existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
                    logger.info("Web app connected to shared memory for video feed.")

                shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=existing_shm.buf)
                frame = shared_frame_array.copy()

                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except FileNotFoundError:
                if existing_shm:
                    existing_shm.close()
                    existing_shm = None
                # Check if placeholder image is available before yielding
                if camera_unavailable_image_bytes:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + camera_unavailable_image_bytes + b'\r\n')
                else:
                    # If placeholder also failed to load, yield nothing or a simple error
                    pass 
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in video feed generator: {e}")
                time.sleep(1)

            time.sleep(0.033)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # ... (end of video_feed function) ...


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
                
                filtered_objects = []
                if analysis.objects:
                    for obj in analysis.objects:
                        # Only include objects that meet the minimum confidence score
                        if obj.confidence >= MIN_OBJECT_CONFIDENCE:
                            filtered_objects.append(obj.as_dict())

                data = {
                    "description": analysis.description.as_dict() if analysis.description else {}, 
                    "tags": [t.as_dict() for t in analysis.tags] if analysis.tags else [], 
                    "objects": filtered_objects,
                }
                
                # --- NEW: Draw bounding boxes on the frame for saving ---
                # NOTE: frame_to_analyze comes from the shared memory access block above
                frame_with_boxes = draw_bounding_boxes(frame_to_analyze.copy(), filtered_objects) # .copy() ensures safety
                ret_box, jpeg_box_bytes = cv2.imencode('.jpg', frame_with_boxes)
                frame_b64 = base64.b64encode(jpeg_box_bytes.tobytes()).decode('utf-8')
                
                # --- NEW: Save the analysis and image to InfluxDB ---
                save_analysis_to_influx(frame_b64, data)
                        
                result_queue.put({"status": "completed", "analysis_data": data})
            except Exception as ex:
                logger.error(f"Analysis task failed: {ex}")
                result_queue.put({"status": "failed", "message": str(ex)})

        result_queue = queue.Queue()
        # Pass the original frame bytes to the worker thread
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

@app.route('/analysis_history', methods=['GET'])
def get_analysis_history():
    """Retrieves the last 5 analysis records (timestamps, descriptions) from InfluxDB."""
    if not query_api:
        return jsonify({"status": "error", "message": "InfluxDB not configured."}), 500
        
    # FLUX query to retrieve the last 5 records
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -30d) 
      |> filter(fn: (r) => r._measurement == "analysis_record")
      |> group(columns: ["_time"])
      |> last() 
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 5)
    '''
    
    records = []
    try:
        result = query_api.query(query=query, org=INFLUXDB_ORG)
        
        # InfluxDB returns one table per field, so we need to collect all fields for the same time.
        
        # 1. Map time to data structure
        time_to_data = {}
        for table in result:
            for record in table.records:
                timestamp = record.get_time()
                field = record.get_field()
                value = record.get_value()
                
                if timestamp not in time_to_data:
                    time_to_data[timestamp] = {}
                time_to_data[timestamp][field] = value

        # 2. Convert to list of history records
        for timestamp, data in time_to_data.items():
            records.append({
                "timestamp": timestamp.isoformat().replace('+00:00', 'Z'), # Format time for JS Date object
                "description": data.get("description", "No description available."),
                "object_count": data.get("object_count", 0)
            })

        # Final sort since `last()` might mess up the global order slightly
        records.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({"status": "success", "history": records})

    except Exception as e:
        logger.error(f"Error querying InfluxDB history: {e}")
        return jsonify({"status": "error", "message": f"Failed to retrieve history: {e}"}), 500
        
@app.route('/analysis_frame/<path:timestamp>', methods=['GET'])
def get_analysis_frame(timestamp):
    """Retrieves the full analysis frame (Base64) for a given timestamp."""
    if not query_api:
        return jsonify({"status": "error", "message": "InfluxDB not configured."}), 500
        
    try:
        # Query all fields for the exact timestamp
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {timestamp}, stop: {timestamp})
          |> filter(fn: (r) => r._measurement == "analysis_record")
          |> last()
        '''
        
        result = query_api.query(query=query, org=INFLUXDB_ORG)
        
        frame_data = {}
        for table in result:
            for record in table.records:
                frame_data[record.get_field()] = record.get_value()
                
        frame_b64 = frame_data.get("frame_b64")
            
        if frame_b64:
            # We return the full metadata here too for a rich modal display
            response_data = {
                "frame_b64": frame_b64,
                "description": frame_data.get("description", "N/A"),
                "object_count": frame_data.get("object_count", 0),
                "tags": json.loads(frame_data.get("tags_json", "[]")),
                "objects": json.loads(frame_data.get("objects_json", "[]")),
            }
            return jsonify({"status": "success", "data": response_data})
        else:
            return jsonify({"status": "error", "message": "Frame not found for that timestamp."}), 404
            
    except Exception as e:
        logger.error(f"Error retrieving frame from InfluxDB: {e}")
        return jsonify({"status": "error", "message": f"Failed to retrieve frame: {e}"}), 500
