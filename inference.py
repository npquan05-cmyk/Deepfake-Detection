import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def predict_image(model, img_path, threshold=0.4):

    img = cv2.imread(img_path)

    if img is None:
        print("Cannot read:", img_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected:", img_path)
        return

    x,y,w,h = faces[0]

    pad = int(0.2*w)

    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(img.shape[1], x+w+pad)
    y2 = min(img.shape[0], y+h+pad)

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face,(224,224)) / 255.0
    face = np.expand_dims(face, axis=0)

    prob = model.predict(face)[0][0]

    label = "REAL" if prob > threshold else "FAKE"
    color = "green" if label=="REAL" else "red"

    print("Image:", os.path.basename(img_path))
    print("Probability REAL:", prob)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} ({prob:.3f})", color=color)
    plt.axis("off")
    plt.show()


def predict_folder(model, folder_path, threshold=0.4):

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png','.jpg','.jpeg','.webp')):
            print("===================================")
            predict_image(model, os.path.join(folder_path, img_name), threshold)