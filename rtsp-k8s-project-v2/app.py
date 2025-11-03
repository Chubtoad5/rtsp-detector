import os
import time
import json
import logging
import multiprocessing.shared_memory as shared_memory
import cv2
import numpy as np
import base64
import io
from urllib.parse import urlparse, urlunparse
from datetime import datetime, timedelta # Added for history modal fix

from flask import Flask, render_template, Response, jsonify, request
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime as dt

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
RTSP_STREAM_URL = os.environ.get("RTSP_STREAM_URL")

# --- InfluxDB Config (From k8s-manifest.yaml Secret) ---
INFLUXDB_URL = os.environ.get("INFLUXDB_URL")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET")

# --- Create Sanitized RTSP URL for Frontend ---
SANITIZED_RTSP_URL = "Not Configured"
if RTSP_STREAM_URL:
    try:
        # Try to parse and remove user/pass
        parsed = urlparse(RTSP_STREAM_URL)
        # Rebuild URL without netloc user/pass
        netloc_parts = parsed.netloc.split('@')
        sanitized_netloc = netloc_parts[-1] # Get the part after '@', or the whole thing
        SANITIZED_RTSP_URL = urlunparse((
            parsed.scheme,
            sanitized_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
    except Exception:
        SANITIZED_RTSP_URL = "rtsp://... (Stream Configured)"


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
    return render_template('index.html', 
                           min_confidence=MIN_OBJECT_CONFIDENCE,
                           rtsp_url=SANITIZED_RTSP_URL) # Pass sanitized URL

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
        # --- FIX for Field Limit Error (as before) ---
        # 1. Query for all STRING fields *EXCEPT* the large frame_b64 field
        query_strings = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -30d) 
          |> filter(fn: (r) => r._measurement == "analysis_record" and 
                             r._field != "object_count" and 
                             r._field != "frame_b64")
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
        
        # Filter out any incomplete records
        history_list = [
            data for data in history_list
            if 'description' in data and 'object_count' in data
        ]
        
        # Sort by timestamp descending
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(history_list[:10]) # Return top 10 recent complete records

    except Exception as e:
        logger.error(f"Failed to query InfluxDB for history: {e}")
        return jsonify({'error': f"Failed to retrieve history: {str(e)}"}), 500

@app.route('/get_historical_record/<path:timestamp>', methods=['GET'])
def get_historical_record(timestamp):
    """Retrieves the full data record (including frame) for a specific timestamp."""
    if not query_api:
        return jsonify({'error': 'InfluxDB client is not ready.'}), 503

    try:
        # --- FIX #1: Timestamp lookup robustness ---
        # 1. Parse the incoming ISO 8601 timestamp string into a Python datetime object.
        # This handles the complexity of the timezone offset (+00:00).
        dt_obj = datetime.fromisoformat(timestamp)
        dt_obj_minus_1s = dt_obj - timedelta(seconds=1)
        dt_obj_plus_1s = dt_obj + timedelta(seconds=1)
        
        # 2. Define a small time range (e.g., +/- 1 second) around the timestamp.
        # This handles small precision differences between the query time and the stored time.
        start_time = dt_obj_minus_1s.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        end_time = dt_obj_plus_1s.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        # 3. Query the range and filter by measurement
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))
          |> filter(fn: (r) => r._measurement == "analysis_record")
          |> filter(fn: (r) => r._time == time(v: "{timestamp}")) # Added for stricter filter
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> limit(n: 1)
        '''
        
        tables = query_api.query(query, org=INFLUXDB_ORG)
        
        if not tables or not tables[0].records:
            logger.warning(f"No record found for timestamp: {timestamp}")
            return jsonify({'error': 'No record found for that timestamp.'}), 404

        # Reconstruct the single record from the pivoted table
        record_data = tables[0].records[0].values
        
        # We only need the fields we're going to display
        historical_record = {
            # Use the record's time for the timestamp
            'timestamp': record_data.get('_time', '').isoformat(), 
            'description': record_data.get('description', ''),
            'tags_json': record_data.get('tags_json', '[]'),
            'objects_json': record_data.get('objects_json', '[]'),
            'object_count': record_data.get('object_count', 0),
            'frame_b64': record_data.get('frame_b64', '')
        }
        
        return jsonify(historical_record)

    except ValueError:
        logger.error(f"Invalid timestamp format received: {timestamp}")
        return jsonify({'error': 'Invalid timestamp format provided.'}), 400
    except Exception as e:
        logger.error(f"Failed to query InfluxDB for specific frame: {e} (Timestamp: {timestamp})")
        return jsonify({'error': f"Failed to retrieve frame: {str(e)}"}), 500

if __name__ == '__main__':
    # This is for local development only
    app.run(host='0.0.0.0', port=8000, debug=True)
