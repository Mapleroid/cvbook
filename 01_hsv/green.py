import cv2
import numpy as np

f="1.jpg"
img=cv2.imread(f)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_blue=np.array([35,100,46])
upper_blue=np.array([77,255,255])

mask=cv2.inRange(hsv,lower_blue,upper_blue)
res=cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('res',res)

cv2.waitKey(0)
