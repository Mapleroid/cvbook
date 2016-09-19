import numpy as np
import cv2
from matplotlib import pyplot as plt

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

def onTMinChanged(x):
    t=cv2.getTrackbarPos('T(Max)','image')
    if x >= t:
        cv2.setTrackbarPos('T(Min)','image',t-1)

def onTMaxChanged(x):
    t=cv2.getTrackbarPos('T(Min)','image')
    if x <= t:
        cv2.setTrackbarPos('T(Max)','image',t+1)

cv2.namedWindow('image')
cv2.createTrackbar('T(min)','image',100,1550,onTMinChanged)
cv2.createTrackbar('T(max)','image',120,1550,onTMaxChanged)

img0 = cv2.imread('4.jpg')
hsv=cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
res = mask(img0,hsv,[0,100,0],[179,255,255])


def intersect(rho1,theta1,rho2,theta2):
    if theta1 == theta2:
        return None

    x = (rho1*np.sin(theta2)-rho2*np.sin(theta1))/np.sin(theta2-theta1)
    y = (rho1*np.cos(theta2)-rho2*np.cos(theta1))/np.sin(theta1-theta2)
    return np.array([x,y])

while(1):  
    k=cv2.waitKey(1)&0xFF 
    if k==27: 
        break

    t_min=cv2.getTrackbarPos('T(min)','image')
    t_max=cv2.getTrackbarPos('T(max)','image')
    img=img0.copy()
    hsv=cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
    res = mask(img0,hsv,[0,100,0],[179,255,255])
    dst=cv2.GaussianBlur(res,(13,13),0)
    #dst=cv2.GaussianBlur(res,(5,5),0)
    edges = cv2.Canny(dst,100,120)
    #cv2.imshow("edge",edges)
    lines = cv2.HoughLines(edges,1,np.pi/30,45)
    

    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 +1000*(-b))
            y1 = int(y0 +1000*(a))
            x2 = int(x0 -1000*(-b))
            y2 = int(y0 -1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
            
    inter_sects = []
    for i in range(0,len(lines[0])-1):
        for j in range(i+1,len(lines[0])):
            rho1,theta1 = lines[0][i]
            rho2,theta2 = lines[0][j]
            inter_sect = intersect(rho1,theta1,rho2,theta2)
            if inter_sect is not None:
                inter_sects.append(inter_sect)

    all_points = np.vstack(inter_sects)
    break
            
    #plt.scatter(lines[0][:,0],lines[0][:,1])
    #plt.xlabel('rho')
    #plt.ylabel('theta')
    #plt.show()

    #cv2.imshow("image",img)

img2 = img0.copy()
for pos in all_points:
    cv2.circle(img2,(pos[0],pos[1]),1,(0,0,255))

cv2.imshow("image",img)
cv2.imshow("image2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#dst=cv2.GaussianBlur(res,(5,5),0)


#imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)




'''
ret,thresh = cv2.threshold(edges,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3) 
'''

#cv2.waitKey(0)