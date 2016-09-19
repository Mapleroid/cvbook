import numpy as np
import cv2
from matplotlib import pyplot as plt

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

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

def get_clusters(lines):
    v_cluster = []
    r_cluster = []
    l_cluster = []

    for rho,theta in lines:
        if (theta >= 0.0 and theta < np.pi/6) or (theta > np.pi*5/6 and theta < np.pi):
            v_cluster.append((rho,theta))

        if (theta > np.pi/5 and theta < np.pi/2-0.1):
            l_cluster.append((rho,theta))

        if (theta > np.pi/2+0.1 and theta < np.pi*3/4):
            r_cluster.append((rho,theta))

    return (v_cluster, r_cluster, l_cluster)

def get_all_lines(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    res = mask(img,hsv,[0,100,0],[179,255,255])
    dst=cv2.GaussianBlur(res,(13,13),0)
    edges = cv2.Canny(dst,100,120)
    lines = cv2.HoughLines(edges,1,np.pi/30,45)

    return lines[0]

def get_clusters_of_cube_by_img(img):
    lines = get_all_lines(img)

    if lines is not None:
        return get_clusters(lines)

def get_clusters_of_cube_by_lines(lines):
    if lines is not None:
        return get_clusters(lines)


def get_boundary_line(cluster):
    pass


def get_center(cluster):
    pass


# test
img = cv2.imread('img/3.jpg')
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()

while(1):  
    k=cv2.waitKey(1)&0xFF 
    if k==27: 
        break

    lines = get_all_lines(img)
    for rho, theta in lines:
        (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
        cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)

    v_cluster, r_cluster, l_cluster = get_clusters_of_cube_by_lines(lines)

    for rho, theta in v_cluster:
        (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
        cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

    for rho, theta in r_cluster:
        (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
        cv2.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

    for rho, theta in l_cluster:
        (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
        cv2.line(img4,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow("img",img1)
    cv2.imshow("v",img2)
    cv2.imshow("r",img3)
    cv2.imshow("l",img4)
    break

cv2.waitKey(0)
cv2.destroyAllWindows()
