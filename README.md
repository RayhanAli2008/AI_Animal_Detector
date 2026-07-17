# AnimalGuessr & Computer Vision Toolkit

This repository contains a collection of computer vision tools designed for real-time animal classification and Augmented Reality (AR) image target feature extraction.

## Project Structure

The project is organized into three main subfolders based on functionality:

### 1. Vuforia image
Contains tools for testing and extracting AR feature tracking points.
* **File:** Vuforia.py
* **Description:** Simulates how AR engines like Vuforia analyze image targets. It uses OpenCV and the ORB (Oriented FAST and Rotated BRIEF) algorithm to convert an image to grayscale and map high-contrast corner points. This script is highly useful for visualizing the "fingerprint" of an image to determine if it makes a good, stable AR tracking target.
* **Usage:** python Vuforia.py (Ensure you have a target image defined in the script, like test_target.png, located in the same folder).

### 2. Webcam Image
Contains the real-time webcam inference scripts.
* **Files:** Webcam.py, oldcode.py
* **Description:** Opens a live webcam feed and processes frames through a pre-trained TensorFlow/Keras model (animalGuesser.keras). It classifies the animal in view from a list of 90 possible classes. To optimize performance, it processes frames at a set interval and overlays the prediction and confidence score directly onto the live video UI.
* **Usage:** python Webcam.py (Press the 'q' key to quit the webcam feed).

### 3. Image
Contains the FastAPI backend for static image classification.
* **File:** Image.py
* **Description:** A local web server that allows users to upload static images via a /predict POST endpoint. It runs the same TensorFlow classification model on the uploaded image. If the prediction confidence score is 70% or higher, it draws a green bounding box and an annotated label on the image, returning the modified visual to the user.
* **Usage:** 1. Run the server: python Image.py
  2. The API will boot up at http://127.0.0.1:8000. 
  3. You can interactively test the upload endpoint by visiting http://127.0.0.1:8000/docs in your browser.

---

## Requirements

To run all modules in this project, you will need the following Python libraries installed:

* opencv-python (cv2)
* tensorflow
* numpy
* fastapi
* uvicorn
* python-multipart (Required for FastAPI file uploads)

You can install all dependencies via pip:
pip install opencv-python tensorflow numpy fastapi uvicorn python-multipart

## Setup Notes
* **Model Pathing:** The scripts in the Webcam Image and Image folders explicitly look for the pre-trained model file animalGuesser.keras at a specific absolute path. Be sure to update the model_path variable in Webcam.py and Image.py to point to the correct location of your .keras file before running them.
