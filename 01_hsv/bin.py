import numpy as np
import cv2

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

img = cv2.imread('1.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
res = mask(img,hsv,[0,100,0],[179,255,255])
dst = cv2.GaussianBlur(res,(5,5),0)
edges = cv2.Canny(dst,100,200)
lines = cv2.HoughLines(edges,1,np.pi/30,45)

#for rho,theta in lines[0]:
#    print "rho:%f, theta:%f " % (rho,theta)

def draw_line(img,rho,theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 +1500*(-b))
    y1 = int(y0 +1500*(a))
    x2 = int(x0 -1500*(-b))
    y2 = int(y0 -1500*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)



draw_line(img,-420.000000, 2.60000)
draw_line(img,540.000000, 0.000000)
#draw_line(img,590.000000, -0.10000)
#draw_line(img,)
#draw_line(img,)
#draw_line(img,)
'''
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
draw_line(img,
'''

cv2.imshow("image",img)
cv2.waitKey(0)
