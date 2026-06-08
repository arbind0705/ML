# Face Recognition System

## Overview

This project is a real-time Face Recognition System built using Python, OpenCV, and the Face Recognition library.

The system captures video from a webcam, detects faces in real time, compares them with a pre-stored reference image, and displays the recognized person's name on the screen.

## Features

* Real-time webcam face detection
* Face recognition using facial encodings
* Automatic face matching
* Bounding boxes around detected faces
* Name display for recognized users
* Handles unknown faces

## Technologies Used

* Python
* OpenCV
* Face Recognition
* NumPy

## How It Works

1. Load a reference image.
2. Generate facial encodings for the reference image.
3. Capture live video from the webcam.
4. Detect faces in each frame.
5. Generate encodings for detected faces.
6. Compare detected faces with known encodings.
7. Display the recognized name and bounding box.

## Installation

```bash
pip install opencv-python
pip install face-recognition
pip install numpy
```

## Usage

1. Place the reference image in the project folder.
2. Update the image path in the code if needed.
3. Run:

```bash
python facedection_system.py
```

4. Press **Q** to exit.

## Future Improvements

* Multiple face registration
* Face attendance system
* Face database integration
* Emotion detection
* Face recognition logging system

## Author

Arbind Malava
