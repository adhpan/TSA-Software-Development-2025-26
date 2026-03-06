# ML Object Classifier using YOLOv11 with TTS

OVERVIEW:
This application detects basic household objects using your webcam and classifies them into a general category by saying the name of the category out loud. Not all objects are able to be classified with accuracy and may be mistaken for other items, as this ML model is pre-trained on just 80 categories.

PURPOSE:
To experiment with AI/ML on a real hands-on project. Primarily built to compete in Technology Student Association (TSA), in the Software Development category.

PIPELINE:
Upon clicking the start/stop button, the webcam receives input, as a singular picture frame. YOLO then processes the frame and makes predictions. A simple mathematical algorithm then calculates the largest/most important object detected. A TTS function then runs with the largest object's name as a parameter, so it can be processed into audio. The TTS function then outputs that audio theough the computer's speakers. Meanwhile, the processed frame with bounding boxes around each detected object is converted in a multi-step process into a usable Tkinter format to be displayed on the application interface. This process then repeats very quickly (30 fps), until the start/stop button is clicked again.
