import os
import time
import json
import logging
import multiprocessing.shared_memory as shared_memory
import cv2
import numpy as np
import base64

from flask import Flask, render_template, Response, jsonify, request
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import datetime as dt # Added for timedelta

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

# 1. Initialize Shared Memory
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=shm.buf)
    logger.info("Web app connected to shared memory for video feed.")
except FileNotFoundError:
    logger.error("Shared memory block not found. Is camera_manager running?")
    shared_frame_array = None

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
        confidence = obj['confidence']
        # Only draw if confidence meets the client-side threshold
        if confidence >= MIN_OBJECT_CONFIDENCE:
            box = obj['rectangle']
            # Scale coordinates to fit the 720x1280 frame
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            
            # Draw rectangle (color BGR: Blue)
            cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Put label and confidence text
            label = f"{obj['object_property']} {confidence:.0%}"
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
        point.field("description", analysis_data.get('description', ''))
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
    if shared_frame_array is None or cv_client is None:
        return {'error': 'Service not initialized.'}

    # 1. Capture the current frame from shared memory
    current_frame = shared_frame_array.copy()

    # 2. Encode frame to memory buffer (JPEG format)
    _, buffer = cv2.imencode('.jpg', current_frame)
    image_bytes = buffer.tobytes()

    # 3. Call Azure Computer Vision
    try:
        analysis_features = ['Description', 'Tags', 'Objects']
        
        # Use a dummy read operation to trigger the API call with the image_bytes stream
        analysis = cv_client.analyze_image_in_stream(image_bytes, analysis_features, language="en")
        
        # 4. Filter objects by confidence score (Server-side filtering)
        filtered_objects = [
            obj for obj in analysis.objects if obj.confidence >= MIN_OBJECT_CONFIDENCE
        ]

        # 5. Compile results
        results = {
            'description': analysis.description,
            'tags': analysis.tags,
            # Convert SDK objects to serializable dicts
            'objects': [{'object_property': obj.object_property, 'confidence': obj.confidence, 'rectangle': obj.rectangle.as_dict()} for obj in filtered_objects]
        }
        
        # 6. Prepare the frame for saving (with boxes drawn for history)
        frame_with_boxes = draw_bounding_boxes(current_frame, results['objects'])
        _, buffer_boxes = cv2.imencode('.jpg', frame_with_boxes)
        frame_b64 = base64.b64encode(buffer_boxes.tobytes()).decode('utf-8')
        
        # 7. Save to InfluxDB asynchronously (or synchronously for simplicity here)
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
    return render_template('index.html', min_confidence=MIN_OBJECT_CONFIDENCE)

def generate_frames():
    """Generator function that yields frames from shared memory."""
    if shared_frame_array is None:
        logger.error("Cannot stream: Shared memory not connected.")
        return

    while True:
        try:
            # Copy the current frame from shared memory
            frame = shared_frame_array.copy()
            
            # Encode frame to JPEG format
            _, buffer = cv2.imencode('.jpeg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Simple rate limiting for streaming
            time.sleep(1/30) 

        except Exception as e:
            logger.error(f"Error streaming frame: {e}")
            break

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
        # 1. Filter out the integer field 'object_count' from the main query 
        #    to guarantee only string values are returned in the _value column.
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -30d) 
          |> filter(fn: (r) => r._measurement == "analysis_record" and r._field != "object_count")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 50) 
        '''
        
        tables = query_api.query(query, org=INFLUXDB_ORG)
        
        # Dictionary to aggregate all fields for a single timestamp
        time_to_data = {}
        
        for table in tables:
            for record in table.records:
                ts = record.values['_time'].isoformat()
                
                if ts not in time_to_data:
                    time_to_data[ts] = {'timestamp': ts}
                
                # Assign the field value based on its key
                field_name = record.values['_field']
                field_value = record.values['_value']
                
                time_to_data[ts][field_name] = field_value

        # 2. Get the integer 'object_count' records separately
        count_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -30d) 
          |> filter(fn: (r) => r._measurement == "analysis_record" and r._field == "object_count")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 50) 
        '''
        count_tables = query_api.query(count_query, org=INFLUXDB_ORG)
        
        # Merge object_count back into time_to_data
        for table in count_tables:
            for record in table.records:
                ts = record.values['_time'].isoformat()
                if ts in time_to_data:
                    time_to_data[ts]['object_count'] = record.values['_value']


        # Convert the aggregated dictionary values into a sorted list
        # Filter out incomplete records (those missing the frame or description)
        history_list = [
            data for ts, data in time_to_data.items()
            if 'description' in data and 'frame_b64' in data
        ]
        
        # Sort by timestamp descending
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(history_list)

    except Exception as e:
        logger.error(f"Failed to query InfluxDB for history: {e}")
        return jsonify({'error': f"Failed to retrieve history: {str(e)}"}), 500

@app.route('/get_historical_frame/<timestamp>', methods=['GET'])
def get_historical_frame(timestamp):
    """Retrieves the full data record for a specific timestamp."""
    if not query_api:
        return jsonify({'error': 'InfluxDB client is not ready.'}), 503

    try:
        # Convert timestamp string back to datetime object to use in Flux query range
        ts_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Define a narrow time window around the timestamp
        start_time = ts_dt.strftime('%Y-%m-%dT%H:%M:%S.%NZ')
        # Use timedelta from the imported dt library
        end_time = (ts_dt + dt.timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%S.%NZ')
        
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {start_time}, stop: {end_time}) 
          |> filter(fn: (r) => r._measurement == "analysis_record" and r._time == {start_time})
        '''
        
        tables = query_api.query(query, org=INFLUXDB_ORG)
        
        # Reconstruct the single record
        historical_record = {'timestamp': timestamp}
        
        for table in tables:
            for record in table.records:
                field_name = record.values['_field']
                field_value = record.values['_value']
                
                # Explicitly cast integer fields back to int if needed
                if field_name == 'object_count':
                    historical_record[field_name] = int(field_value)
                else:
                    historical_record[field_name] = field_value
        
        if 'frame_b64' not in historical_record:
            return jsonify({'error': 'Historical record not found or incomplete.'}), 404

        # Successfully found and reconstructed the single record
        return jsonify(historical_record)

    except Exception as e:
        logger.error(f"Failed to query InfluxDB for specific frame: {e}")
        return jsonify({'error': f"Failed to retrieve frame: {str(e)}"}), 500
