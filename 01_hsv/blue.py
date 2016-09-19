import cv2
import numpy as np

f="5.jpg"
img=cv2.imread(f)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#res = mask(img,hsv,[0,100,0],[179,255,255])

lower_blue=np.array([100,100,46])
upper_blue=np.array([124,255,255])
mask=cv2.inRange(hsv,lower_blue,upper_blue)
res=cv2.bitwise_and(img,img,mask=mask)
dst=cv2.GaussianBlur(res,(5,5),0)

cv2.imshow('res',dst)

cv2.waitKey(0)
