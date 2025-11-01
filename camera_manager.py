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
