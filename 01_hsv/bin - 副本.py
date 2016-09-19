import numpy as np
import cv2

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)


img = cv2.imread('3.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

res = mask(img,hsv,[0,100,0],[179,255,255])
dst=cv2.GaussianBlur(res,(13,13),0)
edges = cv2.Canny(dst,100,120)

#imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("edge",edges)

ret,thresh = cv2.threshold(edges,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img,contours,-1,(0,0,255),3) 


approx = []
for cnt in contours:
    if cv2.contourArea(cnt)> 100:
        approx.append(cnt)

all_points = np.vstack(approx)
hull = cv2.convexHull(all_points)

M = cv2.moments(hull)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
cv2.circle(img,(cx,cy),3,(0,0,255))
for i in range(0,hull.shape[0]-1):
    cv2.line(img,(hull[i,0,0],hull[i,0,1]),(hull[i+1,0,0],hull[i+1,0,1]),(0,0,255),2)

#cv2.drawContours(img,approx,-1,(0,0,255),3) 

cv2.imshow("img",img)
cv2.waitKey(0)