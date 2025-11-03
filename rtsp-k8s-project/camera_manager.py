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
# CAMERA_SOURCE and CONFIG_FILE_PATH are no longer needed

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

# get_camera_source() function is no longer needed

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
        try:
            if not RTSP_STREAM_URL:
                raise ValueError("RTSP_STREAM_URL is not set.")
                
            camera_path = RTSP_STREAM_URL
            camera = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)

            if not camera or not camera.isOpened():
                raise IOError(f"Failed to open camera for source: {camera_path}")
            
            logger.info(f"Camera opened successfully ({camera_path}). Starting frame capture.")

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
