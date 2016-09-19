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
corners = cv2.goodFeaturesToTrack(edges,100,0.01,10)

corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('dst',img)
cv2.waitKey(0)