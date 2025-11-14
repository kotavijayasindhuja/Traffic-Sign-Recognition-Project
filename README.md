Traffic-Sign-Recognition-Project

Real-time traffic sign detection using YOLOv4 and CNN.

This project implements a real-time traffic sign detection and recognition system using:

YOLOv4 for detecting traffic signs

CNN for classifying detected signs

OpenCV for video processing

Text-to-Speech (TTS) for voice alerts

The system identifies important signs such as Stop, Speed Limit 50, Yield, No Entry, Turn Left, and provides live voice warnings.

🚦 Project Overview

Traffic sign recognition plays a major role in autonomous driving and driver-assistance systems.
This project can:

Detect traffic signs from webcam/video

Classify them using a CNN model

Announce detected signs via voice

Work even in foggy or blurred conditions

Compared to models that use only detection or only classification, this combined architecture achieves higher accuracy, real-time performance, and better user interaction.

🧠 Key Features

✔ Real-time traffic sign detection
✔ CNN-based classification
✔ Text-to-speech voice alerts
✔ Fog & blur testing mode
✔ Live webcam support (PC) / simulated frame in Colab
✔ Lightweight and fast model

📂 Project Structure
Traffic-Sign-Recognition-YOLOv4-CNN/
│
├── README.md
├── requirements.txt
├── traffic_cnn.h5
├── yolov4.cfg
├── yolov4.weights
├── coco.names
│
├── dataset/
│    ├── Stop/
│    ├── Speed_Limit_50/
│    ├── Yield/
│    ├── No_Entry/
│    └── Turn_Left/
│
├── src/
│    ├── train_cnn.py
│    ├── realtime_detection_yolo_cnn.py
│    └── dataset_generator.py
│
├── screenshots/
│    ├── Output1.png
│    ├── Output2.png
│    └── Output3.png
│
└── documentation/
     └── Project_Report.pdf

🗃 Dataset Description

The dataset contains five categories of traffic signs:

Stop

Speed Limit 50

Yield

No Entry

Turn Left

Each class includes multiple images for training and testing.

A synthetic dataset generator is provided for Google Colab or environments without external datasets.

🏗 Technologies Used

Python

TensorFlow / Keras

OpenCV

YOLOv4

NumPy

Pyttsx3 (Text-to-Speech)

🧪 How to Run the Project
1. Install Required Packages
pip install -r requirements.txt

2. Train the CNN Model
python src/train_cnn.py

3. Run Real-Time Traffic Sign Recognition
python src/realtime_detection_yolo_cnn.py

⚠ Note for Google Colab Users

Colab cannot access physical webcams.
A dummy frame is used for demonstration instead of a real webcam feed.

📸 Screenshots
<img width="1817" height="825" alt="Screenshot 1" src="https://github.com/user-attachments/assets/9a3d80ff-c4f6-4dd0-b732-3d80dd861f34" /> <img width="1784" height="822" alt="Screenshot 2" src="https://github.com/user-attachments/assets/5cf06a6b-cd02-48bc-99ed-021e2dd9f61e" />
📈 Future Scope

Upgrade to YOLOv7 / YOLOv8

Deploy on Raspberry Pi / Jetson Nano

Add night-mode enhancement

Integrate lane detection & vehicle tracking

Build a mobile app

Train on the GTSRB real-world dataset

👥 Team Members

2300080013 – E. Anoohya

2300080143 – K. Vijaya Sindhuja

2300080204 – N. Ravi Thrayini

2300080278 – G. Durga Sravanthi
