# ML Object Classifier using YOLOv11 with TTS

OVERVIEW:

This application detects basic household objects using your webcam and classifies them into a general category by saying the name of the category out loud.


PURPOSE:

To experiment with AI/ML on a real hands-on project. Primarily built to compete in Technology Student Association (TSA) 2025-2026, in the Software Development category.


PIPELINE:

Upon clicking the start/stop button, the webcam receives input, as a singular picture frame. YOLO then processes the frame and makes predictions. A simple mathematical algorithm then calculates the largest/most important object detected. A TTS function then runs with the largest object's name as a parameter, so it can be processed into audio. The TTS function then outputs that audio through the computer's speakers. Meanwhile, the processed frame with bounding boxes around each detected object is converted in a multi-step process into a usable Tkinter format to be displayed on the application interface. This process then repeats very quickly (30 fps), until the start/stop button is clicked again.


FUNCTIONALITY:

Not all objects are able to be classified with accuracy and may be mistaken for other items, as this ML model is pre-trained on just 80 categories. The program may detect the object as something else, or may not detect it at all.


OUTPUT:

This program utilizes your webcam camera and your speakers, so make sure both are functioning. Your webcam will be running a live feed that you are able to see, and your speakers will be outputting audio every 3 seconds based on whatever the main object is.


DEPENDENCIES: 

opencv-python (for webcam usage), ultralytics (for YOLOv11 ML model), pillow (for Python Imaging Library to process frames), edge-tts (for TTS usage), pygame (for audio engine usage)


TERMINAL COMMAND (RUN BEFORE STARTING):

pip install opencv-python ultralytics pillow edge-tts pygame


