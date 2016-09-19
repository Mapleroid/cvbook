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

def get_distance_to_axisX(rho,theta,3=200):
    return (rho-a*np.cos(theta))/np.sin(theta)


# set trackbar
cv2.namedWindow('image')
img0 = cv2.imread('1.jpg')
while(1):  
    k=cv2.waitKey(1)&0xFF 
    if k==27: 
        break

    hsv=cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
    res = mask(img0,hsv,[0,100,0],[179,255,255])
    dst=cv2.GaussianBlur(res,(13,13),0)
    edges = cv2.Canny(dst,100,120)
    lines = cv2.HoughLines(edges,1,np.pi/30,45)
    
    if lines is not None:
        for rho,theta in lines[0]:
            if (theta >=float(0)/1000 and theta < float(500)/1000) or (theta >float(2700)/1000 and theta <= float(3140)/1000):      
                d = get_distance_to_axisY(rho,theta)
                if d > dist_max:
                    dist_max,rhos_max,theta_max = d, rho, theta

                if d < dist_min:
                    dist_min,rhos_min,theta_min = d, rho, theta


(x1,y1),(x2,y2) = polar2cartesian(rhos_max,theta_max)
cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

(x1,y1),(x2,y2) = polar2cartesian(rhos_min,theta_min)
cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
