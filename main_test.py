import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import math

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/numbers_model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("model/numbers_labels.txt", "r") as f:
    labels = f.read().splitlines()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size == 0:
            continue  # Skip if crop is invalid

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prepare image for TFLite
        imgInput = cv2.resize(imgWhite, (224, 224))  # Teachable Machine default input size
        imgInput = imgInput.astype(np.float32) / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        # Set tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        index = np.argmax(output_data)
        prediction_label = labels[index]
        confidence = output_data[index]

        print("Prediction:", prediction_label, "| Confidence:", confidence)

        # Show result
        cv2.putText(img, f"{prediction_label} ({confidence*100:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

        # Display windows
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)