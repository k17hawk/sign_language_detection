"""
author @ kumar dahal
"""
from flask import Flask, render_template, redirect, url_for,request,jsonify

import cv2
import os
import tensorflow as tf

import mediapipe as mp
import numpy as np
import time
import pandas as pd
from pathlib import Path


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact-us')
def contact():
    return render_template('contact.html')

@app.route('/team')
def team():
    return render_template('Team.html')


@app.route('/handle_button')
def prediction():
    model_dir = 'artifacts/training/model.h5'
    model_path = Path(model_dir)
    model = tf.keras.models.load_model(model_path)
    
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    while True:
        _, frame = cap.read()
        h, w, c = frame.shape

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Extract and preprocess the hand region
                analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe, (28, 28))
                analysisframe = analysisframe / 255.0
                analysisframe = analysisframe.reshape(-1, 28, 28, 1)

                # Make predictions
                prediction = model.predict(analysisframe)
                predarray = np.array(prediction[0])
                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                predicted_character = max(letter_prediction_dict, key=letter_prediction_dict.get)

                # Display the predicted character on the frame
                cv2.putText(frame, f"Predicted Character: {predicted_character}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()
    


@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8002)    