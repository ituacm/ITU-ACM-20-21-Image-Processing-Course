# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:36:45 2020

@author: evren
"""

import cv2
import numpy as np
import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = "0"

#%% Template Matching

source = cv2.imread("./sources/source.jpg")
template= cv2.imread("./sources/template.jpg")

source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w,h = template_gray.shape[::-1]

R = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.6

loc = np.where(R > threshold)

for pt in zip(*loc[::-1]):  #zip ve pt nin ne oldugunu goster
    cv2.rectangle(source, pt,(pt[0] + w, pt[1] + h), (255,120,15), 2)

cv2.imshow("Source",source)
cv2.imshow("Template", template)
#cv2.imshow("R", R)  #bunu da sonra goster R yı yazdıktan sonra


cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Yuz tespiti

img = cv2.imread("./sources/evren.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_xml = cv2.CascadeClassifier("./sources/haarcascade_frontalface_default.xml")

result = face_xml.detectMultiScale(img_gray,1.1, 4)

x,y,w,h = [result[0][i] for i in range(len(result[0]))]
cv2.rectangle(img,(x,y),(x+w,y+h), (255,255,0),3)


cv2.imshow("Evren",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%Yuz tespiti toplu

crowd = cv2.imread("./sources/crowd.jpg")
crowd_gray = cv2.cvtColor(crowd, cv2.COLOR_BGR2GRAY)

faces_xml = cv2.CascadeClassifier("./sources/haarcascade_frontalface_default.xml")

result2 = faces_xml.detectMultiScale(crowd_gray,1.1, 4)


for x,y,w,h in result2:   
    cv2.rectangle(crowd,(x,y),(x+w,y+h), (255,255,0),3)


cv2.imshow("Crowded",crowd)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Anlik Goruntu


cam = cv2.VideoCapture(0)
face_xml = cv2.CascadeClassifier("./sources/haarcascade_frontalface_default.xml") #haarcascade_smile.xml dene

while(cam.isOpened()):
    
    ret,frame = cam.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    my_result = face_xml.detectMultiScale(frame_gray,1.3,4) #yuzde 30 buyuterek goster en az 4 kere yuzu teyit et
    
    for x,y,w,h in my_result:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
    
    cv2.imshow("Frame",frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#%% Hem yuz hem de goz tespiti


cam = cv2.VideoCapture(0)
face_xml = cv2.CascadeClassifier("./sources/haarcascade_frontalface_default.xml")
eyes_xml = cv2.CascadeClassifier("./sources/haarcascade_eye.xml")

while(cam.isOpened()):
    
    _,frame = cam.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    my_result = face_xml.detectMultiScale(frame_gray,1.3,4) #yuzde 30 buyuterek goster en az 4 kere yuzu teyit et
    
    for x,y,w,h in my_result:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
        face_roi = frame[y:y+h,x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes_result = eyes_xml.detectMultiScale(face_roi_gray)
        for x_2,y_2,w_2,h_2 in eyes_result:
            cv2.rectangle(face_roi, (x_2,y_2), (x_2+w_2, y_2 + h_2), (255,0,0),3)
    
    cv2.imshow("Frame",frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#%% Full Body


cam = cv2.VideoCapture("./sources/people.mp4") #people2.mp4 e de bak
face_xml = cv2.CascadeClassifier("./sources/haarcascade_fullbody.xml")

while(cam.isOpened()):
    
    ret,frame = cam.read()
    
    if ret != True:
        print("Video is over...")
        break
    
    frame = cv2.resize(frame,(600,400))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    my_result = face_xml.detectMultiScale(frame_gray,1.3,4) #Rakamlari gittikce dusur oyle dene (ikinci videoda 1.1, 1 yap)
    
    for x,y,w,h in my_result:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
    
    cv2.imshow("Frame",frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#%% Plate Recognition


crowd = cv2.imread("./sources/plate1.jpg") #1,2,3,4 ayri ayri dene hepsini
crowd_gray = cv2.cvtColor(crowd, cv2.COLOR_BGR2GRAY)

#faces_xml = cv2.CascadeClassifier("./sources/haarcascade_licence_plate_rus_16stages.xml")
faces_xml = cv2.CascadeClassifier("./sources/haarcascade_russian_plate_number.xml")


result2 = faces_xml.detectMultiScale(crowd_gray,1.1, 4) #Plate 2 de 1 yap!!


for x,y,w,h in result2:   
    cv2.rectangle(crowd,(x,y),(x+w,y+h), (0,255,255),3)
    

#en son goster (belki) #Plate 1 de goster
plate_roi = crowd[result2[0][1]:result2[0][1]+result2[0][3],result2[0][0]:result2[0][0]+result2[0][2]] 
plate_roi = cv2.resize(plate_roi, (300,200))   
_,thresh = cv2.threshold(plate_roi, 75,255, cv2.THRESH_BINARY)

cv2.imshow("ROI", plate_roi)
cv2.imshow("Thresh", thresh)


cv2.imshow("Crowded",crowd)


cv2.waitKey(0)
cv2.destroyAllWindows()

