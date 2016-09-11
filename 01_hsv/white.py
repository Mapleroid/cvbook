import cv2
import numpy as np

f="2.jpg"
img=cv2.imread(f)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_blue=np.array([0,0,120])
upper_blue=np.array([170,60,255])

mask=cv2.inRange(hsv,lower_blue,upper_blue)
res=cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('res',res)

cv2.waitKey(0)
