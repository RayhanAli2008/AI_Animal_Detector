Animal Guesser - Live Feed is a real-time computer vision application that uses a deep learning model to identify animals from a live webcam feed 
featuring live detection, optimized performance via frame-skipping to keep the video stream smooth, and a readable UI

```
Before running the script, ensure you have the required libraries installed by running:
    `pip install numpy tensorflow opencv-python`
    
For setup, update the model_path variable in the script to point to your specific .keras file location, such as:
    `model_path = r"C:\Users\Rayhan\Downloads\animalGuesser.keras"`
    
Note that the script defaults to your primary webcam cv2.VideoCapture(0) but can be changed to 1 for an external webcam. 

To use the application, run the script using:
    `python animal_guesser.py`
    
This will open the "Animal Guesser - Live Feed" window. Point your camera at an animal to see the prediction and confidence percentage,
 and press the 'q' key on your keyboard to safely exit.

```
