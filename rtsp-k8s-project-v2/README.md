# RTSP Object Detector - v2 Deployment Guide

This project is now configured with InfluxDB history. Follow these "cache-busting" steps **exactly** to deploy the new version and guarantee your cluster runs the new code.

## 1. Add Placeholder Image

- **Manually** add your `camera_unavailable.jpg` file to the `static/` folder.

## 2. Update Placeholders

You **MUST** edit these files and replace the placeholders:

1.  **`k8s-influxdb.yaml`**:
    - `YOUR_INFLUXDB_TOKEN`: Set a strong, secure token here.

2.  **`k8s-manifest.yaml`**:
    - `YOUR_AZURE_KEY_HERE`: Set your Azure key.
    - `https.your-endpoint.cognitiveservices.azure.com/`: Set your Azure endpoint.
    - `rtsp://your-stream-url-here`: Set your RTSP stream URL.
    - `YOUR_INFLUXDB_TOKEN`: Use the *same token* from step 1.
    - `http://rtsp-detector.your-domain.com`: Set your public URL.
    - `your-registry/rtsp-object-detector:v1`: **IMPORTANT!** Update the `your-registry` part to your Docker registry (e.g., `docker.io/yourusername`).

## 3. Build & Push the **Versioned** Image

This is the most critical step. We are using a `v1` tag, not `:latest`.

```bash
# Navigate to this project directory
cd rtsp-k8s-project-v2

# 1. Build the image with a NEW tag
docker build -t your-registry/rtsp-object-detector:v1 .

# 2. Push the image
docker push your-registry/rtsp-object-detector:v1
```

## 4. Deploy to Kubernetes

```bash
# 1. Deploy InfluxDB first and wait for it to be 'Running'
kubectl apply -f k8s-influxdb.yaml
kubectl get pods -w

# 2. Deploy the application
kubectl apply -f k8s-manifest.yaml
```

## 5. Verify

Because you used a new tag (`:v1`), Kubernetes is **forced** to pull the new image.

- Check the logs of the `web-app` container. You should see it initialize the InfluxDB client.
- Open your browser and do a **hard refresh** (`Ctrl+Shift+R` or `Cmd+Shift+R`).
- The "Analyze" and "History" buttons should now work.

## Updating in the Future

If you make new code changes, build and push with a **new** tag (e.g., `:v2`), update the `image:` line in `k8s-manifest.yaml`, and run `kubectl apply` again.
