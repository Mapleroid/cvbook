import cv2
import numpy as np

# 640*480
cap=cv2.VideoCapture(0)


vertexs = (
    (105,50),
    (320,5),
    (560,75),
    (530,350),
    (320,480),
    (115,325)
)
vertex0 = (330,150)

while(1):
    ret,frame=cap.read()

    for i in range(0,5):
        cv2.line(frame,vertexs[i],vertexs[i+1],(0,0,255),2)
    cv2.line(frame,vertexs[5],vertexs[0],(0,0,255),2)

    for i in range(0,6,2):
        cv2.line(frame,vertex0,vertexs[i],(0,0,255),2)


    cv2.imshow('frame',frame)

    k=cv2.waitKey(5)&0xFF
    if k==27:
        break

cv2.destroyAllWindows()