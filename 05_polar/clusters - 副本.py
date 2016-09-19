import numpy as np
import cv2
from matplotlib import pyplot as plt

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

def nothing(x):
    pass

def polar2cartesian(rho,theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 +1000*(-b))
    y1 = int(y0 +1000*(a))
    x2 = int(x0 -1000*(-b))
    y2 = int(y0 -1000*(a))

    return (x1,y1),(x2,y2)

def get_distance_to_axisY(rho,theta,a=200):
    return (rho-a*np.sin(theta))/np.cos(theta)

def get_distance_to_axisX(rho,theta,a=200):
    return (rho-a*np.cos(theta))/np.sin(theta)

def cluster_lines(lines):
    
    v_cluster = []
    r_cluster = []
    l_cluster = []

    for rho,theta in lines:
        if (theta >= 0.0 and theta < 0.5) or (theta > np.pi*5/6 and theta < np.pi):
            v_cluster.append((rho,theta))

        if (theta > np.pi/4 and theta < np.pi/2-0.1):
            r_cluster.append((rho,theta))

        if (theta > np.pi/2+0.1 and theta < np.pi*3/4):
            l_cluster.append((rho,theta))

    return (v_cluster, r_cluster, l_cluster)



img0 = cv2.imread('1.jpg')
cv2.namedWindow('image')

while(1):  
    k=cv2.waitKey(1)&0xFF 
    if k==27: 
        break

    #img=img0.copy()
    img2=img0.copy()
    img3=img0.copy()
    img4=img0.copy()
    hsv=cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
    res = mask(img0,hsv,[0,100,0],[179,255,255])
    dst=cv2.GaussianBlur(res,(13,13),0)
    edges = cv2.Canny(dst,100,120)
    lines = cv2.HoughLines(edges,1,np.pi/30,45)
    
    if lines is not None:
        v_cluster, r_cluster, l_cluster = cluster_lines(lines[0])
        
        for rho, theta in v_cluster:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

        for rho, theta in r_cluster:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

        for rho, theta in l_cluster:
            (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
            cv2.line(img4,(x1,y1),(x2,y2),(0,0,255),2)

        #cv2.imshow("image",img)
        cv2.imshow("v",img2)
        cv2.imshow("r",img3)
        cv2.imshow("l",img4)
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
