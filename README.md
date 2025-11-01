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
