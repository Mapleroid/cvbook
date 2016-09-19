import cv2
import numpy as np

f="1.jpg"
img=cv2.imread(f)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

'''
lower_blue=np.array([100,43,46])
upper_blue=np.array([124,255,255])
mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)
res_blue=cv2.bitwise_and(img,img,mask=mask)
'''

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

res_blue = mask(img,hsv,[100,43,46],[124,255,255])
res_green = mask(img,hsv,[35,43,46],[77,255,255])
res_orange = mask(img,hsv,[11,43,46],[25,255,255])
res_red = mask(img,hsv,[0,43,46],[6,255,255])
res_white = mask(img,hsv,[0,0,120],[170,60,255])
res_yellow = mask(img,hsv,[17,165,120],[34,255,255])

res = cv2.add(res_blue,res_green)
res = cv2.add(res,res_orange)
res = cv2.add(res,res_red)
res = cv2.add(res,res_white)
res = cv2.add(res,res_yellow)

cv2.imshow('res_blue',res_blue)
cv2.imshow('res_green',res_green)
cv2.imshow('res_orange',res_orange)
cv2.imshow('res_red',res_red)
cv2.imshow('res_white',res_white)
cv2.imshow('res_yellow',res_yellow)
cv2.imshow('res',res)
cv2.waitKey(0)
