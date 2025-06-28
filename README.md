# SnapChat-Filter
Animated Snapchat-style Filters with MediaPipe
Showcase your creativity by applying real-time animated filters (like smoke on open mouth or glowing eyes) using MediaPipe's face landmark detection and OpenCV.

# Features
->Detects faces using MediaPipe Face Detection

->Extracts 468 facial landmarks in real time with Face Mesh

->Geometric logic to determine if eyes or mouth are open

->Overlays dynamic filters:

>Animated smoke when mouth is open
    
>Custom PNG filters on open eyes

->Smooth real-time webcam performance (60 + FPS on modern CPUs/GPU)

# Setup
->Install dependencies:

    pip install opencv-python mediapipe matplotlib numpy
 
->Create a media/ folder containing:
 >left_eye.png, right_eye.png (transparent overlays)
 
 >smoke_animation.mp4 (looping smoke/video file)
 
 >Test images: sample.jpg, sample2.jpg, sample3.jpg

# How It Works
->Core Functions
detectFacialLandmarks(image, face_mesh, display=True)
Runs face mesh detection, draws tessellation and contours, and returns annotated image and results.

python:
results = face_mesh.process(image[..., ::-1])
drawing logic...
return annotated, results

getSize(image, face_landmarks, INDEXES)
Computes the bounding box width, height, and landmarks for a specific face region (e.g., eye or mouth).

isOpen(image, face_mesh_results, face_part, threshold, display=True)
Determines if eyes/mouth are open by comparing part height to face height, outputs annotated image and status.

python:
status = {face_id: "OPEN" if ratio > threshold else "CLOSE"}
cv2.putText(...)

overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True)
Centers and scales your filter graphic over the target region (scaling to ~2.5 × detected height), applying proper masking to preserve transparency.

->Real-time Loop (Webcam + Filters)
 ->>Opens webcam via OpenCV (VideoCapture(0) or 2)
 ->>Syncs animated smoke using frame counters and resets at end-of-loop
 ->>For each frame:
  ->>>Flip for selfie view
  ->>>Detect landmarks
  ->>>Check open-state of mouth/eyes
  ->>>Overlay filters conditionally
  ->>>Show live feed with cv2.imshow() and FPS label

# Usage
Run:
main.py

Use ESC to exit real-time filter mode.

# Static Image Demos
Within your script, call:

python:
detectFacialLandmarks(cv2.imread("media/sample2.jpg"), face_mesh_images)
isOpen(...), overlay(...)

To visualize filter effects on sample images.

# Acknowledgements
Built upon concepts from the tutorial "Facial Landmark Detection with MediaPipe – Creating Animated Snapchat Filters" by Bleed AI Academy.

