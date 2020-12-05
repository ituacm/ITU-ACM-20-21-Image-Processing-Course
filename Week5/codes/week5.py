# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:57:45 2020

@author: evren
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#%% Sobel

sudoku = cv2.imread("./images/sudoku.png")

cv2.imshow("Sudoku",sudoku)
sudoku_gray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)

laplace = cv2.Laplacian(sudoku_gray, cv2.CV_32F)
sobelx = cv2.Sobel(sudoku_gray, cv2.CV_32F, 1,0, ksize=5)
sobely = cv2.Sobel(sudoku_gray, cv2.CV_32F, 0,1, ksize=5)

cv2.imshow("SobelX", sobelx)
cv2.imshow("SobelY", sobely)
cv2.imshow("Laplacian", laplace)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%Canny Edge

img = cv2.imread("./images/ground.jpg",0) 
#img = cv2.resize(img,(500,750))

canny_edge = cv2.Canny(img, 800,900)

cv2.imshow("Ground", img)
cv2.imshow("Canny Edge", canny_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Histogram Equalization


img = cv2.imread("./images/very_dark.jpg")
img = cv2.resize(img,(600,400))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.equalizeHist(img_gray)


output = np.hstack((img_gray,hist))

cv2.imshow("Output", output)


plt.figure(figsize=(10,6.5))

plt.subplot(2,1,1)
plt.hist(img_gray.ravel(),256,[0,256])
plt.title("Non Equalized Image")

plt.subplot(2,1,2)
plt.hist(hist.ravel(),256,[0,256])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Color Range 

def nothing(x):
    #any operation
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower-Hue", "Trackbars", 0,180, nothing)
cv2.createTrackbar("Lower-Saturation", "Trackbars", 66,255, nothing)
cv2.createTrackbar("Lower-Value", "Trackbars", 134,255, nothing)
cv2.createTrackbar("Upper-Hue", "Trackbars", 180,180, nothing)
cv2.createTrackbar("Upper-Saturation", "Trackbars", 255,255, nothing)
cv2.createTrackbar("Upper-Value", "Trackbars", 243,255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX



while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    u_h = cv2.getTrackbarPos("Upper-Hue", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper-Saturation", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper-Value", "Trackbars")
    l_h = cv2.getTrackbarPos("Lower-Hue", "Trackbars")
    l_s = cv2.getTrackbarPos("Lower-Saturation", "Trackbars")
    l_v = cv2.getTrackbarPos("Lower-Value", "Trackbars")

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    


#%%Erosion Dilation

img = cv2.imread("./images/acm_noise_2.png",0)
img = cv2.resize(img,(600,400))

kernel = np.ones((5,5), np.uint8)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


cv2.imshow("Normal", img)
cv2.imshow("Opening", closing)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Renk tespiti

cam = cv2.VideoCapture(0)
lower_thresh = (56,107,69)
upper_thresh = (147,255,217)

kernel = np.ones((5,5), np.uint8)

while True:
    _,frame = cam.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = cv2.bitwise_and(mask,mask, mask = mask)
    
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#%%


cam = cv2.VideoCapture(0)
lower_thresh = (56,107,69)
upper_thresh = (147,255,217)
kernel = np.ones((5,5), np.uint8)

while(True):
    ret,frame = cam.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = cv2.bitwise_and(mask, mask, mask=mask)
  
  	# Masking
    mask = cv2.erode(mask, kernel, iterations=4)
    mask = cv2.dilate(mask, kernel, iterations=4)
    #mask = cv2.GaussianBlur(mask, (5, 5), 0)
    canny_img = cv2.Canny(mask, 700, 900)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,255,0),3)
    
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    
    height, width = canny_img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Rectangle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    
    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
        cv2.putText(frame, "Rectangle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()









