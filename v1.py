import tkinter as tk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
from time import time
from edge_tts import Communicate
import asyncio
from threading import Thread
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import get_player_name

# Main window
root = tk.Tk()
root.title("Developers: Adhrit Pantam, Vedant Patil")
root.geometry("1000x850")
root.config(bg="#000000")

# Headings
padding = tk.Label(root)
padding.pack(pady=10)
padding.config(bg="#000000")

intro = tk.Label(root)
intro.pack(pady=5)
intro.config(bg="#000000", fg="#ffffff", text="Technology Student Association - Software Development - 2025-26", font=("Times New Roman", 16))

intro2 = tk.Label(root)
intro2.pack(pady=5)
intro2.config(bg="#000000", fg="#ffffff", text="ML Object Classifier with TTS", font=("Times New Roman", 14))

# Webcam display
display = tk.Label(root)
display.pack(pady=30)
display.config(bg="#000000", fg="#ffffff", text="This application detects basic household objects using your webcam and classifies them into a general category by saying the name of the category out loud.\n" \
"Not all objects are able to be classified with accuracy and may be mistaken for other items, as this ML model is pre-trained on just 80 categories.")

object = tk.Label(root)
object.pack(pady=10)
object.config(bg="#000000", fg="#ffffff", font=("Times New Roman", 1))

# Load ML model
model = YOLO("yolo11m.pt")

webcam = None
running = False
last_said = ""
last_said_time = 0

# Default to ffplay
get_player_name = lambda: "ffplay"

# Run TTS
async def say_prediction(main_object):
    tts = Communicate(text="A " + main_object, voice="en-US-AriaNeural")
    audio_buffer = bytearray()
    async for sample in tts.stream():
        if sample['type'] == 'audio':
            audio_buffer.extend(sample['data'])
    audio = AudioSegment.from_file(BytesIO(audio_buffer), format="mp3")
    play(audio)

# Initiate TTS
def run_tts(main_object):
    asyncio.run(say_prediction(main_object))

#Main webcam function
def run_webcam():
    global webcam, running, last_said, last_said_time
    if running:
        # Retrieve webcam frame
        success, frame = webcam.read()
        if not success:
            return
        
        # Get prediction
        predictions = model(frame)
        result = predictions[0]

        # Calculate main object based on largest area
        ids = list(result.boxes.cls)
        coords = list(result.boxes.xyxy)
        max_area = 0
        main_object = ""
        for i in range(len(ids)): 
            id = ids[i]
            x1 = coords[i][0]
            y1 = coords[i][1]
            x2 = coords[i][2]
            y2 = coords[i][3]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area: 
                max_area = area
                main_object = result.names[int(id)]

        # Run TTS if conditions met
        if main_object != "" and time() - last_said_time > 3:
            last_said = main_object
            last_said_time = time()
            thread = Thread(target=run_tts, args=(main_object,), daemon=True).start()

        # Convert frame to Tkinter-friendly picture and display
        frame_w_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_w_color)
        imgtk = ImageTk.PhotoImage(img)
        display.config(image=imgtk)
        object.config(text=main_object)
        display.reference = imgtk # Prevent garbage collection

        # Repeat function for next frame
        root.after(30, run_webcam)

# Button functionality
def start_stop_running():
    global webcam, running, last_said, last_said_time
    if running:
        last_said = ""
        last_said_time = 0
        webcam.release()
        display.config(image='')
        object.config(text='', font=("Times New Roman", 1))
        running = False
    else:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            return
        object.config(font=("Times New Roman", 16))
        running = True
        run_webcam()

# Button
start = tk.Button(root, text="Start/Stop Webcam", command=start_stop_running)
start.pack(pady=20, ipady=5, ipadx=10)
start.config(bg="#211929", fg="#f0e8fa", font=("Times New Roman", 12))

root.mainloop()

