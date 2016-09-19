import numpy as np
import cv2

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

img = cv2.imread('1.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
res = mask(img,hsv,[0,100,0],[179,255,255])
dst = cv2.GaussianBlur(res,(5,5),0)
edges = cv2.Canny(dst,100,200)


#gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
gray = np.float32(edges)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
#img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)

cv2.imshow("image",dst)
cv2.waitKey(0)