import cv2
import cvzone
import math
import streamlit as st
from ultralytics import YOLO
from sort import *

st.set_page_config(
        page_title="Helmet detection",
        page_icon="🤖",
        layout="wide",
        menu_items={
            'About': "Helmet detection project."
        }
    )

st.title("Helmet Detection Stream")

video_source = st.selectbox("Select video source", ["sample.mp4"])

run_btn = st.button("Start Detection")

FRAME_WINDOW = st.image([])

if run_btn:
    cap = cv2.VideoCapture(0 if video_source == "Webcam" else video_source)
    cap.set(3, 1280)
    cap.set(4, 720)

    model = YOLO('best (1).pt')
    classNames = ['numberPlate', 'NoHelmet', 'GoodHelmet', 'BadHelmet', 'rider']
    myColor = (0, 0, 255)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.warning("Failed to read from video source.")
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "GoodHelmet":
                    myColor = (0, 255, 0)
                elif currentClass in ["NoHelmet", "BadHelmet"]:
                    myColor = (0, 0, 255)
                else:
                    myColor = (255, 255, 255)

                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1,
                                   colorB=myColor, colorT=(255, 0, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)

        FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
