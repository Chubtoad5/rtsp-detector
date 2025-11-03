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
