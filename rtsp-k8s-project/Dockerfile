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
