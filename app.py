import torch
import cv2
import os
import io
import pyttsx3
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, url_for, redirect, Response
import random
import smtplib
from email.message import EmailMessage
import sqlite3
import pythoncom
from object_info import object_descriptions  # 👈 for object details




# Initialize Flask
app = Flask(__name__)

from flask import session
app.secret_key = 'Yafiah@2005'  # Add this below app = Flask(__name__)

# Initialize text-to-speech


# Define the correct YOLOv5 model path
MODEL_PATH = r"C:\Users\anjum\OneDrive\Desktop\FINEGRAINED - Copy_pre1\yolov5-master\best.pt"

# Ensure YOLOv5 is properly set up
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load YOLOv5 model correctly
model = torch.hub.load(r"C:\Users\anjum\OneDrive\Desktop\FINEGRAINED - Copy_pre1\yolov5-master", 
                        "custom", 
                        path=r"C:\Users\anjum\OneDrive\Desktop\FINEGRAINED - Copy_pre1\yolov5-master\best.pt", 
                        source="local")
model.to("cpu")  # Change to "cuda" if you have a GPU

# Function to process video frames
def gen():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            img = Image.fromarray(frame)
            results = model(img, size=640)  # Run inference
            img = np.squeeze(results.render())  # Get image with bounding boxes
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

import threading

def speak_text(text):
    pythoncom.CoInitialize()
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    pythoncom.CoUninitialize()

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Please select an image!")

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Perform YOLOv5 detection
        results = model(img, size=640)
        detections = results.pandas().xyxy[0]

        object_counts = {}
        for obj in detections["name"]:
            object_counts[obj] = object_counts.get(obj, 0) + 1  # Count occurrences

        detected_text = ", ".join([f"{obj} ({count})" for obj, count in object_counts.items()])

        object_info_list = []
        tts_lines = []

        if object_counts:
            detected_names = list(object_counts.keys())
            for obj in detected_names:
                desc = object_descriptions.get(obj, "No description available.")
                object_info_list.append({"name": obj, "description": desc})
                tts_lines.append(f"{obj}: {desc}")

            full_tts = "Detected objects: " + ", ".join(detected_names) + ". " + ". ".join(tts_lines)
            tts_thread = threading.Thread(target=speak_text, args=(full_tts,))
            tts_thread.start()

        # Render image
        print("Before rendering: ", img.size)
        rendered_img = np.squeeze(results.render())
        print("After rendering: ", rendered_img.shape)
        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)

        # Save image
        import time
        if not os.path.exists("static"):
            os.makedirs("static")
        unique_filename = f"detected_image_{int(time.time() * 1000)}.jpg"
        output_path = os.path.join("static", unique_filename)
        cv2.imwrite(output_path, rendered_img)

        # Save data for assistant
        session["object_info_list"] = object_info_list
        session["object_counts"] = object_counts
        session["image_url"] = url_for("static", filename=unique_filename)
        session["detected_text"] = detected_text

        return render_template(
        "detected.html",
        image_url=session["image_url"],
        detected_text=detected_text,
        object_info_list=object_info_list,
        object_counts=object_counts
)


    return render_template("index.html")


@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/logon')
def logon():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('signin.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        mail1 = request.form.get("user", "")
        password1 = request.form.get("password", "")

        con = sqlite3.connect("signup.db")
        cur = con.cursor()
        cur.execute("SELECT `user`, `password` FROM info WHERE `user` = ? AND `password` = ?", (mail1, password1,))
        data = cur.fetchone()
        con.close()

        if data:
            return redirect(url_for("index"))  # Redirect after successful login
        else:
            return render_template("signin.html", error="Invalid username or password")

    return render_template("signin.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/notebook")
def notebook():
    return render_template("Notebook.html")

if __name__ == "__main__":
    app.run(debug=True)