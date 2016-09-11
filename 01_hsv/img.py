import cv2
import numpy as np

f="1.jpg"
img=cv2.imread(f)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
(h,s,v) = cv2.split(hsv)
#hsv2=cv2.merge(h,s)
cv2.imshow("H",h) 
cv2.imshow("S",s) 
cv2.imshow("V",v) 
#cv2.imshow("HSV2",hsv2)
#cv2.imshow(f,img)
cv2.waitKey(0)

