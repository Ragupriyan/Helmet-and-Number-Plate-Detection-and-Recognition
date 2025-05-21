import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *

cap = cv2.VideoCapture('bike detector.mp4')

cap.set(3 , 1280)
cap.set(4 , 720)

model = YOLO('best (1).pt')

classNames = ['numberPlate' , 'NoHelmet' ,'GoodHelmet' , 'BadHelmet' , 'rider']
myColor = (0,0,255)

while True:
    success , img = cap.read()
    results = model(img , stream = True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
            w , h = x2-x1 , y2-y1
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            #cvzone.putTextRect(img , f'{conf}' , (max(0,x1) , max(35,y1)) )

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == "GoodHelmet"):
                myColor = (0,255,0)
            elif (currentClass == "NoHelmet" or currentClass == "BadHelmet"):
                myColor = (0,0,255)
            else:
                myColor = (255,255,255)

            cvzone.putTextRect(img , f'{classNames[cls]} {conf}' ,
                               (max(0,x1) , max(35,y1)) , scale = 1 , thickness = 1,
                               colorB = myColor , colorT=(255,0,255) , colorR = myColor , offset = 5)
            cv2.rectangle(img , (x1 ,y1) , (x2,y2) , myColor , 1)


    cv2.imshow('Video' , img)
    cv2.waitKey(1)
