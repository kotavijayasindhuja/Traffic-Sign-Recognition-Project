***SOURCE CODE FOR TRAFFIC SIGN RECOGNITION PROJECT:***





\# full\_pipeline\_train\_and\_webcam.py

import os

import zipfile

import cv2

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers, models, callbacks

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pyttsx3

from PIL import Image, ImageDraw

from google.colab.patches import cv2\_imshow # Explicitly import for Colab



\# ---------------------------

\# 0. PARAMETERS

\# ---------------------------

DATASET\_DIR = "unzipped\_dataset"   # target dataset dir (flow\_from\_directory expects class subfolders)

ZIP\_NAME = "traffic\_dataset.zip"

SYNTHETIC\_PER\_CLASS = 50          # number of synthetic images per class if dataset missing

IMG\_SIZE = (64, 64)

BATCH\_SIZE = 16

EPOCHS = 8                        # change higher for real training

MODEL\_FILENAME = "traffic\_cnn.h5"



YOLO\_CFG = "yolov4.cfg"           # ensure these files exist

YOLO\_WEIGHTS = "yolov4.weights"

COCO\_NAMES = "coco.names"



\# ---------------------------

\# 1. Ensure dataset exists (create small synthetic if not)

\# ---------------------------

classes = \["Stop", "Speed\_Limit\_50", "Yield", "No\_Entry", "Turn\_Left"]



def create\_synthetic\_dataset(out\_dir, classes, per\_class=10):

&nbsp;   os.makedirs(out\_dir, exist\_ok=True)

&nbsp;   def draw\_sign(text, filename):

&nbsp;       img = Image.new("RGB", (200, 200), "white")

&nbsp;       d = ImageDraw.Draw(img)

&nbsp;       d.rectangle(\[12, 12, 188, 188], outline="red", width=12)

&nbsp;       # draw text centered-ish

&nbsp;       d.text((30, 80), text, fill="red")

&nbsp;       img = img.resize((IMG\_SIZE\[0], IMG\_SIZE\[1]))

&nbsp;       img.save(filename, "JPEG", quality=90)



&nbsp;   for cls in classes:

&nbsp;       cls\_dir = os.path.join(out\_dir, cls)

&nbsp;       os.makedirs(cls\_dir, exist\_ok=True)

&nbsp;       for i in range(per\_class):

&nbsp;           draw\_sign(cls, os.path.join(cls\_dir, f"{cls}\_{i}.jpg"))



if not os.path.exists(DATASET\_DIR) or len(os.listdir(DATASET\_DIR)) == 0:

&nbsp;   print("Dataset not found or empty — creating a synthetic dataset. (Small demo set)")

&nbsp;   create\_synthetic\_dataset(DATASET\_DIR, classes, per\_class=SYNTHETIC\_PER\_CLASS)

&nbsp;   # optionally zip it for record

&nbsp;   with zipfile.ZipFile(ZIP\_NAME, 'w') as z:

&nbsp;       for root, \_, files in os.walk(DATASET\_DIR):

&nbsp;           for f in files:

&nbsp;               z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DATASET\_DIR))

&nbsp;   print("Synthetic dataset created at:", DATASET\_DIR)



\# ---------------------------

\# 2. Prepare data generators

\# ---------------------------

train\_datagen = ImageDataGenerator(

&nbsp;   rescale=1./255,

&nbsp;   validation\_split=0.2,

&nbsp;   rotation\_range=10,

&nbsp;   width\_shift\_range=0.1,

&nbsp;   height\_shift\_range=0.1,

&nbsp;   zoom\_range=0.1,

&nbsp;   horizontal\_flip=False

)



train\_gen = train\_datagen.flow\_from\_directory(

&nbsp;   DATASET\_DIR,

&nbsp;   target\_size=IMG\_SIZE,

&nbsp;   batch\_size=BATCH\_SIZE,

&nbsp;   class\_mode='categorical',

&nbsp;   subset='training',

&nbsp;   shuffle=True

)



val\_gen = train\_datagen.flow\_from\_directory(

&nbsp;   DATASET\_DIR,

&nbsp;   target\_size=IMG\_SIZE,

&nbsp;   batch\_size=BATCH\_SIZE,

&nbsp;   class\_mode='categorical',

&nbsp;   subset='validation',

&nbsp;   shuffle=False

)



\# Save class order

class\_indices = train\_gen.class\_indices

\# invert to label list ordered by index

labels\_by\_index = \[None] \* len(class\_indices)

for label, idx in class\_indices.items():

&nbsp;   labels\_by\_index\[idx] = label

print("Classes:", labels\_by\_index)



\# ---------------------------

\# 3. Build a lightweight CNN

\# ---------------------------

def build\_cnn(input\_shape=(IMG\_SIZE\[0], IMG\_SIZE\[1], 3), num\_classes=len(labels\_by\_index)):

&nbsp;   model = models.Sequential(\[

&nbsp;       layers.Input(shape=input\_shape),

&nbsp;       layers.Conv2D(32, (3,3), activation='relu', padding='same'),

&nbsp;       layers.BatchNormalization(),

&nbsp;       layers.MaxPool2D((2,2)),

&nbsp;       layers.Conv2D(64, (3,3), activation='relu', padding='same'),

&nbsp;       layers.BatchNormalization(),

&nbsp;       layers.MaxPool2D((2,2)),

&nbsp;       layers.Conv2D(128, (3,3), activation='relu', padding='same'),

&nbsp;       layers.BatchNormalization(),

&nbsp;       layers.MaxPool2D((2,2)),

&nbsp;       layers.Flatten(),

&nbsp;       layers.Dense(128, activation='relu'),

&nbsp;       layers.Dropout(0.4),

&nbsp;       layers.Dense(num\_classes, activation='softmax')

&nbsp;   ])

&nbsp;   return model



model = build\_cnn()

model.compile(optimizer='adam', loss='categorical\_crossentropy', metrics=\['accuracy'])

model.summary()



\# ---------------------------

\# 4. Train the CNN

\# ---------------------------

checkpoint\_cb = callbacks.ModelCheckpoint(MODEL\_FILENAME, save\_best\_only=True, monitor='val\_accuracy', mode='max')

earlystop\_cb = callbacks.EarlyStopping(monitor='val\_loss', patience=5, restore\_best\_weights=True)



print("\\nStarting training... (this will take time depending on dataset size \& machine)")

history = model.fit(

&nbsp;   train\_gen,

&nbsp;   validation\_data=val\_gen,

&nbsp;   epochs=EPOCHS,

&nbsp;   callbacks=\[checkpoint\_cb, earlystop\_cb]

)



\# After training, ensure best model is saved

if os.path.exists(MODEL\_FILENAME):

&nbsp;   print("Saved best model to:", MODEL\_FILENAME)

else:

&nbsp;   model.save(MODEL\_FILENAME)

&nbsp;   print("Saved model to:", MODEL\_FILENAME)



\# ---------------------------

\# 5. Load YOLOv4 (for detection)

\# ---------------------------

if not (os.path.exists(YOLO\_CFG) and os.path.exists(YOLO\_WEIGHTS) and os.path.exists(COCO\_NAMES)):

&nbsp;   print("\\nWARNING: One or more YOLO files (cfg/weights/names) not found.")

&nbsp;   print("Place yolov4.cfg, yolov4.weights and coco.names in the script folder to enable YOLO detection.")

&nbsp;   # We'll still run webcam using CNN on full-frame fall-back, but detection won't run.

use\_yolo = os.path.exists(YOLO\_CFG) and os.path.exists(YOLO\_WEIGHTS) and os.path.exists(COCO\_NAMES)



if use\_yolo:

&nbsp;   net = cv2.dnn.readNet(YOLO\_WEIGHTS, YOLO\_CFG)

&nbsp;   with open(COCO\_NAMES, 'r') as f:

&nbsp;       coco\_classes = \[c.strip() for c in f.readlines()]

&nbsp;   layer\_names = net.getLayerNames()

&nbsp;   output\_layers = \[layer\_names\[i\[0] - 1] for i in net.getUnconnectedOutLayers()] if isinstance(net.getUnconnectedOutLayers()\[0], (list, np.ndarray)) else \[layer\_names\[i - 1] for i in net.getUnconnectedOutLayers()]

&nbsp;   print("YOLO loaded, COCO classes:", len(coco\_classes))

else:

&nbsp;   net = None

&nbsp;   coco\_classes = \[]



\# ---------------------------

\# 6. Start webcam -> detect -> classify -> speak

\# ---------------------------

engine = pyttsx3.init()

def speak(text):

&nbsp;   engine.say(text)

&nbsp;   engine.runAndWait()



\# load the trained model (best checkpoint)

cnn\_model = tf.keras.models.load\_model(MODEL\_FILENAME)

print("Loaded trained CNN model.")



cap = cv2.VideoCapture(0)

webcam\_available = cap.isOpened()



if not webcam\_available:

&nbsp;   print("WARNING: Cannot open webcam. Running with a single dummy frame for demonstration.")

&nbsp;   # Create a dummy frame (e.g., black or with some text)

&nbsp;   dummy\_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Or appropriate size

&nbsp;   cv2.putText(dummy\_frame, "No Webcam Detected (Dummy Frame)", (50, 240),

&nbsp;               cv2.FONT\_HERSHEY\_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE\_AA)

&nbsp;   frames\_to\_process = \[dummy\_frame] # Process only one dummy frame

else:

&nbsp;   print("Starting webcam. Press 'q' to quit.")

&nbsp;   frames\_to\_process = \[] # Will be populated by cap.read() in loop



frame\_count = 0

MAX\_DEMO\_FRAMES = 5 # Process a few frames if webcam available, or just one dummy frame



while True:

&nbsp;   if webcam\_available:

&nbsp;       ret, frame = cap.read()

&nbsp;       if not ret:

&nbsp;           break # End of video stream

&nbsp;       # Optional: Limit number of frames from webcam for demo if desired

&nbsp;       frame\_count += 1

&nbsp;       if frame\_count > MAX\_DEMO\_FRAMES:

&nbsp;           print(f"Processed {MAX\_DEMO\_FRAMES} frames from webcam. Stopping demo.")

&nbsp;           break

&nbsp;   else:

&nbsp;       # If webcam not available, we only have the dummy frame to process once

&nbsp;       if not frames\_to\_process: # Already processed the dummy frame

&nbsp;            break

&nbsp;       frame = frames\_to\_process.pop(0) # Get the dummy frame



&nbsp;   frame\_disp = frame.copy()

&nbsp;   h, w = frame.shape\[:2]



&nbsp;   boxes = \[]



&nbsp;   if use\_yolo:

&nbsp;       # run YOLO detection

&nbsp;       blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)

&nbsp;       net.setInput(blob)

&nbsp;       outs = net.forward(output\_layers)



&nbsp;       confidences = \[]

&nbsp;       for out in outs:

&nbsp;           for detection in out:

&nbsp;               scores = detection\[5:]

&nbsp;               class\_id = int(np.argmax(scores)) if len(scores)>0 else -1

&nbsp;               conf = float(scores\[class\_id]) if len(scores)>0 else 0.0

&nbsp;               

&nbsp;               # Accept any detection with high confidence from YOLO

&nbsp;               # The CNN will perform specific traffic sign classification

&nbsp;               if conf > 0.5: # Consider a detection good enough for further CNN processing

&nbsp;                   center\_x = int(detection\[0] \* w)

&nbsp;                   center\_y = int(detection\[1] \* h)

&nbsp;                   bw = int(detection\[2] \* w)

&nbsp;                   bh = int(detection\[3] \* h)

&nbsp;                   x = max(0, center\_x - bw//2)

&nbsp;                   y = max(0, center\_y - bh//2)

&nbsp;                   x2 = min(w-1, int(x + bw))

&nbsp;                   y2 = min(h-1, int(y + bh))

&nbsp;                   

&nbsp;                   # Ensure minimal size to avoid errors with cv2.resize

&nbsp;                   if (x2 - x > 0 and y2 - y > 0):

&nbsp;                       boxes.append(\[x, y, x2-x, y2-y, conf, class\_id]) # Storing x, y, width, height, confidence, original YOLO class\_id

&nbsp;       

&nbsp;       # Non-max suppression

&nbsp;       if boxes:

&nbsp;           nms\_boxes = \[\[b\[0], b\[1], b\[2], b\[3]] for b in boxes] # NMS expects \[x,y,w,h]

&nbsp;           scores = np.array(\[b\[4] for b in boxes])

&nbsp;           

&nbsp;           indexes = cv2.dnn.NMSBoxes(nms\_boxes, list(scores), 0.5, 0.4)

&nbsp;           keep\_indices = \[idx\[0] for idx in indexes] if len(indexes) > 0 else \[]

&nbsp;       else:

&nbsp;           keep\_indices = \[]



&nbsp;       # Process kept boxes

&nbsp;       for idx in keep\_indices:

&nbsp;           x, y, bw, bh, conf\_yolo, class\_id\_yolo = boxes\[idx]

&nbsp;           x2 = x + bw

&nbsp;           y2 = y + bh

&nbsp;           crop = frame\[y:y2, x:x2]

&nbsp;           if crop.size == 0 or crop.shape\[0] == 0 or crop.shape\[1] == 0:

&nbsp;               continue



&nbsp;           # prepare crop for CNN

&nbsp;           crop\_resized = cv2.resize(crop, IMG\_SIZE)

&nbsp;           crop\_resized = crop\_resized.astype('float32') / 255.0

&nbsp;           crop\_resized = np.expand\_dims(crop\_resized, axis=0)

&nbsp;           pred = cnn\_model.predict(crop\_resized, verbose=0)

&nbsp;           pred\_idx = int(np.argmax(pred\[0]))

&nbsp;           label = labels\_by\_index\[pred\_idx]

&nbsp;           prob = float(np.max(pred\[0]))



&nbsp;           # draw box \& label

&nbsp;           cv2.rectangle(frame\_disp, (x, y), (x2, y2), (0,255,0), 2)

&nbsp;           cv2.putText(frame\_disp, f"{label} {prob:.2f}", (x, max(15,y-5)),

&nbsp;                       cv2.FONT\_HERSHEY\_SIMPLEX, 0.6, (0,255,0), 2)



&nbsp;           # speak (optional: only if high confidence)

&nbsp;           if prob > 0.6:

&nbsp;               speak(f"{label} ahead")



&nbsp;   else:

&nbsp;       # Fallback: if YOLO not available, run CNN on centered crop of full frame

&nbsp;       cx, cy = w//2, h//2

&nbsp;       size = min(w,h)//2

&nbsp;       x = cx - size//2

&nbsp;       y = cy - size//2

&nbsp;       crop = frame\[y:y+size, x:x+size]

&nbsp;       crop\_resized = cv2.resize(crop, IMG\_SIZE)

&nbsp;       crop\_resized = crop\_resized.astype('float32') / 255.0

&nbsp;       crop\_resized = np.expand\_dims(crop\_resized, axis=0)

&nbsp;       pred = cnn\_model.predict(crop\_resized, verbose=0)

&nbsp;       pred\_idx = int(np.argmax(pred\[0]))

&nbsp;       label = labels\_by\_index\[pred\_idx]

&nbsp;       prob = float(np.max(pred\[0]))



&nbsp;       cv2.rectangle(frame\_disp, (x,y), (x+size, y+size), (255,0,0), 2)

&nbsp;       cv2.putText(frame\_disp, f"{label} {prob:.2f}", (x, max(15,y-5)),

&nbsp;                   cv2.FONT\_HERSHEY\_SIMPLEX, 0.8, (255,0,0), 2)



&nbsp;       if prob > 0.6:

&nbsp;           speak(f"{label} ahead")



&nbsp;   # Display frame using cv2\_imshow for Colab compatibility

&nbsp;   cv2\_imshow(frame\_disp)



&nbsp;   # In Colab, cv2.waitKey won't work as expected for interactive breaks.

&nbsp;   # The loop will break naturally after processing MAX\_DEMO\_FRAMES or the single dummy frame.

&nbsp;   if not webcam\_available:

&nbsp;       break # Break loop after processing dummy frame



if webcam\_available: # Only release if a webcam was actually opened

&nbsp;   cap.release()

\# cv2.destroyAllWindows() # Not needed for cv2\_imshow

print("Webcam simulation/processing finished.")

