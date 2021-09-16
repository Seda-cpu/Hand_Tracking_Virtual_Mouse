#-*-coding:utf-8-*-
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from time import sleep
from pynput.keyboard import Controller
import time

cap = cv2.VideoCapture(0)


cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

keys = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U'],
        ['V', 'Y', 'Z', 'Q', 'W', 'X', '.', ',', '?', ' ']]

sonuc = ''
keyboard = Controller()
def draw_all(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x+w, y+h), (232, 162, 0), 2, cv2.LINE_AA) 
        cv2.putText(img, button.text, (x +20, y+80), cv2.FONT_HERSHEY_SIMPLEX, 3, (167, 73, 163), 3)
    return img

class Button():
    def __init__(self, pos, text, size=[100,100]):
        self.pos = pos
        self.size = size
        self.text= text
    


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([110*j+70, 100*i+50], key))

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = detector.findHands(frame)
    landmark_list, bbox_info = detector.findPosition(img, draw=False)
    img = draw_all(img, buttonList)
         
    if landmark_list:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < landmark_list[8][0] < x+w and y < landmark_list[8][1] < y+h:
                cv2.rectangle(img, button.pos, (x+w, y+h), (204, 72, 63), 2, cv2.LINE_AA) 
                cv2.putText(img, button.text, (x +20, y+80), cv2.FONT_HERSHEY_SIMPLEX, 3, (167, 73, 163), 3)

                l, _, _ = detector.findDistance(4, 8, img, draw=False)
                #print(l)

                if l<35:
                    keyboard.press(button.text)
                    cv2.rectangle(img, button.pos, (x+w, y+h), (0, 255, 255), 2, cv2.FILLED) 
                    cv2.putText(img, button.text, (x +20, y+80), cv2.FONT_HERSHEY_SIMPLEX, 3, (167, 73, 163), 3)

                    sonuc += button.text
                    sleep(0.2)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.rectangle(img, (60,350), (750,450), (232, 162, 0), 2, cv2.LINE_AA) 
    cv2.putText(img, sonuc, (60,430), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
    cv2.putText(img, 'FPS: {0:.2f}'.format(fps), (1100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2, cv2.LINE_AA)
    cv2.putText(img, 'KIRCI ROBOTICS', (30,700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2,cv2.LINE_AA)

    cv2.imshow('Sanal Klavye', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
