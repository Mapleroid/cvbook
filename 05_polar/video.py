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

def get_distance_to_axisY(rho,theta,a=200):
    return (rho-a*np.sin(theta))/np.cos(theta)

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

# set trackbar
cv2.namedWindow('image')
cv2.createTrackbar('min1','image',0,3140,nothing)
cv2.createTrackbar('max1','image',500,3140,nothing)
cv2.createTrackbar('min2','image',2700,3140,nothing)
cv2.createTrackbar('max2','image',3140,3140,nothing)

#img0 = cv2.imread('3.jpg')

cap=cv2.VideoCapture(0)
while(1): 
    dist_max=0.0
    dist_min=2000.0
    rhos_max=0.0
    rhos_min=2000.0
    theta_max=0.0
    theta_min=0.0
    k=cv2.waitKey(1)&0xFF 
    if k==27: 
        break

    min1=cv2.getTrackbarPos('min1','image')
    max1=cv2.getTrackbarPos('max1','image')
    if min1>max1:
        min1,max1 = max1,min1

    min2=cv2.getTrackbarPos('min2','image')
    max2=cv2.getTrackbarPos('max2','image')
    if min2>max2:
        min2,max2 = max2,min2

    ret,frame=cap.read()
    img=frame.copy()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    res = mask(frame,hsv,[0,100,0],[179,255,255])
    dst=cv2.GaussianBlur(res,(13,13),0)
    edges = cv2.Canny(dst,100,120)
    lines = cv2.HoughLines(edges,1,np.pi/30,45)
    
    if lines is not None:
        for rho,theta in lines[0]:
            #if (theta >= float(min1)/1000 and theta < float(max1)/1000) or (theta > float(min2)/1000 and theta <= float(max2)/1000):
            #if not (
            #    (theta >float(500)/1000 and theta < float(800)/1000) or
            #    (theta > float(1500)/1000 and theta < float(1800)/1000) or
            #    (theta > float(2400)/1000 and theta < float(2700)/1000)
            #    ):
            #if theta >float(800)/1000 and theta < float(1500)/1000:
                #cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)
                #cv2.putText(img2,"%.1f,%.3f" % (rho,theta),(int(100rho),y2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))


            # lines vertical
            
            if (theta >=float(0)/1000 and theta < float(500)/1000) or (theta >float(2700)/1000 and theta <= float(3140)/1000):
                d = get_distance_to_axisY(rho,theta)
                if d > dist_max:
                    dist_max,rhos_max,theta_max = d, rho, theta

                if d < dist_min:
                    dist_min,rhos_min,theta_min = d, rho, theta
                

            #if (theta >=float(650)/1000 and theta < float(1500)/1000):
            #    cv2.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

            #if (theta >=float(1800)/1000 and theta < float(2400)/1000):
            #    cv2.line(img4,(x1,y1),(x2,y2),(0,0,255),2)

        (x1,y1),(x2,y2) = polar2cartesian(rhos_max,theta_max)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        (x1,y1),(x2,y2) = polar2cartesian(rhos_min,theta_min)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("image",img)
    #cv2.imshow("left",img3)
    #cv2.imshow("right",img4)

#cv2.imshow("image",img)
#cv2.imshow("vertical",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
