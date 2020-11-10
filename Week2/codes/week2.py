# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:25:52 2020

@author: evren
"""

#Importing libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Reading an Image

img = cv2.imread("./images/Logo.png")

cv2.imshow("Opencv Logo", img)

cv2.imwrite("./logo_gri.png",img) #Writing an image

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%Some basic features

picture = cv2.imread("./images/Istanbul.jpeg")

#print(picture.shape)

b, g, r=cv2.split(picture)
"""
print("Blue: ", b)
print("Green: ", g)
print("Red: ", r)
"""
my_pixel = picture[220,13,:]

print("Blue: ", my_pixel[0])
print("Green: ", my_pixel[1])
print("Red: ", my_pixel[2])

cv2.imshow("Istanbul", picture)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Converting Image to Gray, HSV ...

scenery = cv2.imread("./images/scenery.jpg")

scenery_gray = cv2.cvtColor(scenery, cv2.COLOR_BGR2GRAY)
scenery_hsv = cv2.cvtColor(scenery, cv2.COLOR_BGR2HSV)
scenery_hls = cv2.cvtColor(scenery, cv2.COLOR_BGR2HLS)
scenery_bgra = cv2.cvtColor(scenery, cv2.COLOR_BGR2BGRA)
scenery_CIE = cv2.cvtColor(scenery, cv2.COLOR_BGR2XYZ)


cv2.imshow("Original", scenery)
cv2.imshow("Gray",scenery_gray)
cv2.imshow("HSV", scenery_hsv)
cv2.imshow("HLS",scenery_hls)
cv2.imshow("BGRA",scenery_bgra)
cv2.imshow("CIE",scenery_CIE)



cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Splitting and showing one channel of image

B,G,R = cv2.split(picture)
picture = cv2.merge((B,G,R))

cv2.imshow("Blue",B)
cv2.imshow("Green",G)
cv2.imshow("Red",R)


cv2.waitKey(0)
cv2.destroyAllWindows()


#%% More Understandable version

deneme_img = cv2.imread("./images/Logo.png")

cv2.imshow("Original",deneme_img)
b,g,r = cv2.split(deneme_img)

cv2.imshow("B",b)
cv2.imshow("G",g)
cv2.imshow("R",r)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Seeing the OpenCV logo as 2 channels

copy1 = deneme_img.copy()
copy2 = deneme_img.copy()
copy3 = deneme_img.copy()

copy1[:,:,2] = 0 #Kırmızı yok
copy2[:,:,1] = 0 #Yesil yok
copy3[:,:,0] = 0 #Mavi yok

cv2.imshow("Original",deneme_img)
cv2.imshow("No red", copy1)
cv2.imshow("No green", copy2)
cv2.imshow("No blue", copy3)


cv2.waitKey(0)
cv2.destroyAllWindows()


