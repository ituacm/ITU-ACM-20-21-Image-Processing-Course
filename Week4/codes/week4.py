# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:48:45 2020

@author: evren
"""


import cv2
import numpy as np
import os

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = "0"

#%% Recording Video

camera = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.mp4", fourcc, 10.0, (640,480))

while(camera.isOpened()):
    
    ret, frame = camera.read()
    
    if ret == True:
        out.write(frame)
        
        cv2.imshow("Video is recording, please press q to stop recording", frame)
        
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    
    else:
        break
    
camera.release()
out.release()
cv2.destroyAllWindows()


#%%Adding operation

x = np.uint8([29])
y = np.uint8([230])

print("Numpy adding: ", np.add(x,y))
print("OpenCV adding: ", cv2.add(x,y))


#%% Image Addinfg



logo = cv2.imread("./images/opencv.png")
zebra = cv2.imread("./images/zebra.jpg")

zebra = cv2.resize(zebra, (324,378))

new_image = cv2.add(logo,zebra)
weighted_image = cv2.addWeighted(logo,0.3,zebra,0.8,0)


cv2.imshow("Opencv", logo)
cv2.imshow("Zebra", zebra)
cv2.imshow("Sum", new_image)
cv2.imshow("Weighted Image", weighted_image)




cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Bitwise Operation

first_img = cv2.imread("./images/input1.png")
second_img = cv2.imread("./images/input2.png")



cv2.imshow("First Image", first_img)
cv2.imshow("Second Image",second_img )

and_operator = cv2.bitwise_and(first_img, second_img, mask = None)
or_operator = cv2.bitwise_or(first_img, second_img, mask = None)
xor_operator = cv2.bitwise_xor(first_img, second_img, mask = None)
not_operator_1 = cv2.bitwise_not(first_img, mask = None)
not_operator_2 = cv2.bitwise_not(second_img, mask = None)


cv2.imshow("And", and_operator)
cv2.imshow("Or", or_operator)
cv2.imshow("XOR", xor_operator)

cv2.imshow("Not Operator 1", not_operator_1)
cv2.imshow("Not Operator 2", not_operator_2)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Masking

logo = cv2.imread("./images/logo.jpg")
blank_chart = cv2.imread("./images/blank_chart.jpg")

rows,cols,channels = logo.shape

roi = blank_chart[75:75+rows,250:250+cols]

logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

ret,mask = cv2.threshold(logo_gray, 25,255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

image_blank = cv2.bitwise_and(roi,roi, mask = mask_inv)

img_logo = cv2.bitwise_and(logo,logo, mask = mask)

dst = cv2.add(image_blank, img_logo)


blank_chart[75:75+rows,250:250+cols] = dst

cv2.imshow("Logo", logo)
cv2.imshow("Blank Chart", blank_chart)
cv2.imshow("lOGO gRAY", logo_gray)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Inv", mask_inv)
cv2.imshow("Image Blank", image_blank)
cv2.imshow("DST", dst)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Bluring Operations

img = cv2.imread("./images/noisy_image.png")


kernel = np.ones((5,5), np.float32)/25
output = cv2.filter2D(img, -1, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Blurred Image", output)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Another blurring


img = cv2.imread("./images/noisy_image_2.png")

output = cv2.blur(img,(11,11))
gaussian_blur = cv2.GaussianBlur(img,(11,11),0)

cv2.imshow("Original Image", img)
cv2.imshow("Blurred Image", output)
cv2.imshow("Gaussian Image", gaussian_blur)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Median Blurring

 
img = cv2.imread("./images/noisy_image_2.png")

median = cv2.medianBlur(img,5)

cv2.imshow("Original Image", img)
cv2.imshow("Median Blur Image", median)


cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Color Filtering Hue Saturation Value 


book = cv2.imread("./images/book.jpg")

book = cv2.resize(book, (1000,720))


hsv = cv2.cvtColor(book, cv2.COLOR_BGR2HSV)

lower_green = (26,77,68)
upper_green = (54,189,255)

mask = cv2.inRange(hsv, lower_green, upper_green)
final = cv2.bitwise_and(book,book, mask = mask)

cv2.imshow("Original Image", book)
cv2.imshow("Masked Image", mask)
cv2.imshow("Final", final)

cv2.waitKey(0)
cv2.destroyAllWindows()



#%% Video Color Filtering


cam = cv2.VideoCapture(0)
lower_green = (21,135,105)
upper_green = (58,255,243)

while(cam.isOpened()):
    ret, frame = cam.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green,upper_green)
    final_frame = cv2.bitwise_and(frame,frame, mask = mask)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Final", final_frame)

    if cv2.waitKey(10)  & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()









