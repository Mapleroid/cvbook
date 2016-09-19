import numpy as np
import cv2
from matplotlib import pyplot as plt

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

def get_center_lines(Z):
    for i in range(0,len(Z)):
        if Z[i][1]<0.5:
            Z[i][1]+= np.pi

    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness,labels,centers = cv2.kmeans(Z[:,1],3,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    A = Z[labels.ravel()==0]
    B = Z[labels.ravel()==1]
    C = Z[labels.ravel()==2]

    ret1,label1,centers1=cv2.kmeans(A,min(7,len(A)),criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    ret2,label2,centers2=cv2.kmeans(B,min(7,len(B)),criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    ret3,label3,centers3=cv2.kmeans(C,min(7,len(C)),criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    return (centers1, centers2, centers3)

def polar2cartesian(rho,theta):
    if theta > np.pi:
        theta -= np.pi

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 +1000*(-b))
    y1 = int(y0 +1000*(a))
    x2 = int(x0 -1000*(-b))
    y2 = int(y0 -1000*(a))

    return (x1,y1),(x2,y2)


img0 = cv2.imread('2.jpg')
while(1):  
    k=cv2.waitKey(1)&0xFF 
    if k==27: 
        break

    img1=img0.copy()
    img2=img0.copy()
    img3=img0.copy()
    img4=img0.copy()
    hsv=cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
    res = mask(img0,hsv,[0,100,0],[179,255,255])
    dst=cv2.GaussianBlur(res,(13,13),0)
    edges = cv2.Canny(dst,100,120)
    lines = cv2.HoughLines(edges,1,np.pi/30,45)
    
    if lines is not None:
        for rho, theta in lines[0]:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
        
        center1, center2, center3 = get_center_lines(lines[0])

        for rho, theta in center1:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

        for rho, theta in center2:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

        for rho, theta in center3:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img4,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("img",img1)
        cv2.imshow("v1",img2)
        cv2.imshow("v2",img3)
        cv2.imshow("v3",img4)
        break

cv2.waitKey(0)
cv2.destroyAllWindows()