import os
import cv2
import numpy as np
import time
import logging
import multiprocessing.shared_memory as shared_memory

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Frame Configuration ---
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
FRAME_CHANNELS = 3
FPS = 30
FRAME_SLEEP_TIME = 1 / FPS # Time to sleep between frame captures

# --- Shared Memory Configuration ---
SHM_NAME = 'object_detector_frame_buffer'
SHM_SIZE = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS

# --- Environment Variables ---
# *** FIX: Changed to RTSP_STREAM_URL to match the variable expected by your application's codebase (as seen in the logs). ***
RTSP_URL = os.environ.get("RTSP_STREAM_URL")
if not RTSP_URL:
    logger.error("RTSP_STREAM_URL environment variable is not set. Exiting.")
    exit(1)

# =============================================================================
# INITIALIZATION
# =============================================================================
# 1. Initialize Shared Memory
try:
    # Try to create shared memory block (only the first process succeeds)
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    logger.info(f"Created Shared Memory: {SHM_NAME} with size {SHM_SIZE}")
except FileExistsError:
    # If it exists, attach to it
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    logger.info(f"Attached to existing Shared Memory: {SHM_NAME}")

# Create a NumPy array backed by shared memory
shared_frame_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8, buffer=shm.buf)

# 2. Camera Initialization (Using GStreamer Pipeline for Stability)

# --- NEW GStreamer Pipeline (More resilient for K8s/RTSP) ---
# rtsp:// protocol, 5-second timeout, TCP transport (more reliable than UDP), 
# then decode the video stream using H264/H265 decoder and convert to BGR format.
# IMPORTANT: This requires GStreamer to be correctly installed in your container base image (e.g., opencv-python on a Linux image).

gstreamer_pipeline = (
    f'rtspsrc location={RTSP_URL} protocols=tcp timeout=5000 '
    '! decodebin ! videoconvert ! videoscale '
    f'! video/x-raw, width={FRAME_WIDTH}, height={FRAME_HEIGHT}, format=(string)BGR '
    '! appsink drop=true'
)

def initialize_camera():
    """Initializes the camera capture object, preferring the GStreamer pipeline."""
    logger.info("Attempting to open RTSP stream using GStreamer pipeline...")
    # cv2.CAP_GSTREAMER (or 700) is the index for GStreamer backend
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        logger.error(f"Failed to open stream with GStreamer. Trying simple URL method.")
        # Fallback to simple URL method
        cap = cv2.VideoCapture(RTSP_URL) 
    
    if not cap.isOpened():
        logger.error(f"FATAL: Could not open video stream from URL: {RTSP_URL}")
        return None

    # Set properties (these may be ignored by RTSP/GStreamer)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    logger.info("Successfully connected to video stream.")
    return cap

# =============================================================================
# MAIN LOOP
# =============================================================================
def run_camera_manager():
    cap = initialize_camera()
    
    if cap is None:
        # Exit if camera initialization failed
        shm.close()
        return

    while True:
        try:
            # Read a frame from the camera
            ret, frame = cap.read()

            if ret:
                # Resize the frame to the target resolution (important for consistent SHM access)
                resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Copy the resized frame data to the shared memory array
                np.copyto(shared_frame_array, resized_frame)
                
            else:
                logger.warning("Failed to read frame from stream. Reconnecting...")
                cap.release()
                cap = initialize_camera()
                if cap is None:
                    # If reconnect fails, wait and try again
                    time.sleep(5)
                
            # Control the frame rate
            time.sleep(FRAME_SLEEP_TIME)

        except KeyboardInterrupt:
            logger.info("Camera Manager stopped by user.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
            time.sleep(1)

    # Cleanup (important in Python to release resources)
    cap.release()
    shm.close()
    # Note: We do NOT unlink the SHM here, as the web app container needs it.
    
if __name__ == "__main__":
    run_camera_manager()
