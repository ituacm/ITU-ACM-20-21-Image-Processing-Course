# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:09:27 2020

@author: evren
"""
#Import Library
import cv2
import os

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = "0"   #For cameras which still on while the code finish execution


#%% Resim bastırma

img = cv2.imread("./images/Logo.png")

cv2.imshow("OpenCV Resim", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%Pixel size

picture = cv2.imread("./images/Istanbul.jpeg")

print(picture.size)
print(picture.shape)
cv2.imshow("Istanbul", picture)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

my_copy = picture.copy()
roi = my_copy[175:350,0:900]

my_copy[350:525,0:900] = roi


cv2.imshow("Istanbul", my_copy)
cv2.imshow("ROI", roi)


cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Resimlerde cerceveleme

import matplotlib.pyplot as plt

color = cv2.imread("./images/Logo2.png")

cv2.imshow("Original Image", color)

green = [0,255,0]
red = [0,0,255]


border1 = cv2.copyMakeBorder(color,15,15,15,15,cv2.BORDER_WRAP)
border2 = cv2.copyMakeBorder(color,15,15,15,15,cv2.BORDER_REFLECT)
border3 = cv2.copyMakeBorder(color,15,15,15,15,cv2.BORDER_REFLECT101)
border4 = cv2.copyMakeBorder(color,15,15,15,15,cv2.BORDER_REPLICATE)
border5 = cv2.copyMakeBorder(color,15,15,15,15,cv2.BORDER_CONSTANT,value = green)
border6 = cv2.copyMakeBorder(color,15,15,15,15,cv2.BORDER_ISOLATED,value =red)

image_dict = {"wrap": border1,"reflect": border2, "reflect101":border3,"replicate":border4,"constant":border5,
              "isolated":border6}

i=1
for key,value in image_dict.items():
    plt.subplot(2,3,i)
    plt.imshow(value)
    plt.title(key)
    plt.axis("off")
    i+=1

plt.show()
    

cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Resolution Changing


resolution = {"width": 200,"height": 100}
pug = cv2.imread("./images/pug.jpg")



new_width = resolution["width"]/pug.shape[1]
new_height = resolution["height"]/pug.shape[0]

scale = min(new_width,new_height)

my_width = int(img.shape[1]*scale)
my_height = int(img.shape[0]*scale)

name="new_window"



cv2.namedWindow(name,cv2.WINDOW_NORMAL)
cv2.resizeWindow(name,my_width, my_height)

cv2.imshow(name, pug)



cv2.imshow("Pug",pug)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Image Pyramids (Gaussian Pyramid)


nature = cv2.imread("./images/nature.jpg")

lower_nature = cv2.pyrDown(nature)
higher_nature = cv2.pyrUp(nature)

cv2.imshow("Higher Nature", higher_nature)
cv2.imshow("Nature", nature)
cv2.imshow("Lower Nature", lower_nature)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Drawing rectangle, circle and put text into the image

acm = cv2.imread("./images/acm.png")

rectangle = acm.copy()
circle = acm.copy()
text = acm.copy()

cv2.imshow("ACM", acm)


starting_point = (550,200)
ending_point = (1360, 700)
border_color = (0,255,255)
thickness = 4 # 4px

rectangle = cv2.rectangle(rectangle, starting_point,ending_point,border_color, thickness)

center = (960,450)
radius = 400
circle = cv2.circle(circle, center, radius, border_color, thickness)

cv2.imshow("Rectangle", rectangle)
cv2.imshow("Circle", circle)


bottom_left_point = (600,750)
my_text = "Welcome to the Image Processing World !!"
text = cv2.putText(text, my_text, bottom_left_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, color = border_color)
cv2.imshow("Text",text)


cv2.waitKey(0)
cv2.destroyAllWindows()




#%% Video running

video = cv2.VideoCapture("./images/video.mp4")

while(True):
    
    ret,frame = video.read()
    
    if ret == False:
        print("Video is over")
        break
    
    frame = cv2.resize(frame,(1280,720))
    cv2.imshow("video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video.release()
cv2.destroyAllWindows()

#%% Live camera

live = cv2.VideoCapture(0)

while(True):
    
    _, live_frame = live.read()
    cv2.imshow("video", live_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
live.release()
cv2.destroyAllWindows()

#%% Color operations

cam = cv2.VideoCapture("./images/grafitti.mp4")
frame_counter = 0
while(True):
    
    ret,frame = cam.read()
    
    frame_counter +=1
    if frame_counter == cam.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter =0
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if ret == False:
        print("Video is over")
        break
    
    frame = cv2.resize(frame, (640,480))
    
    graffiti_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    graffiti_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    graffiti_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    graffiti_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    graffiti_CIE = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
    

    
    cv2.imshow("Original", frame)
    cv2.imshow("Gray", graffiti_gray)
    cv2.imshow("HSV", graffiti_hsv)
    cv2.imshow("HLS", graffiti_hls)
    cv2.imshow("BGRA", graffiti_bgra)
    cv2.imshow("XYZ", graffiti_CIE)


    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()














