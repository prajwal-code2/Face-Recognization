import numpy as np
import face_recognition
import cv2
import datetime
import pyttsx3
import pickle
import os
import streamlit as st

def img_to_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(image)
    embedding = face_recognition.face_encodings(image, locations)
    os.remove(image_path)
    return embedding


def speak(text):
    engine = pyttsx3.init('sapi5')
    engine.setProperty("rate", 200)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[6].id)
    engine.say(text)
    engine.runAndWait()


def wishme(identity=None):
    hour = int(datetime.datetime.now().hour)
    if identity == None:
        if hour >= 0 and hour < 12:
            speak(f"Good morning")
        elif hour >= 12 and hour < 18:
            speak(f"Good afternoon")
        else:
            speak(f"Good evening")
    else:
        if hour >= 0 and hour < 12:
            speak(f"Good morning {identity}")
        elif hour >= 12 and hour < 18:
            speak(f"Good afternoon {identity}")
        else:
            speak(f"Good evening {identity}")


def who_is_it(image_path):
    with open('database.pkl', 'rb') as fp:
        database = pickle.load(fp)
    encoding = img_to_encoding(image_path)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = face_recognition.face_distance(np.array(encoding), np.array(db_enc))
        if dist.any() < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.6:
        wishme()
        speak("May I know your name")
        yourname = takeCommand()
        database[yourname] = encoding
        if "" in database:
            del database[""]
        with open('database.pkl', 'wb') as fp:
            pickle.dump(database, fp)
    else:
        wishme(identity)


def recognize(image_path):
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    cv2.imwrite(image_path, img)
    del (cam)
    cv2.destroyAllWindows()
    who_is_it(image_path)


if __name__ == '__main__':
    recognize('image.jpg')
