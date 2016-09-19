import numpy as np
import cv2
from matplotlib import pyplot as plt

Z = np.array([
[1027.000000, 0.942478 ],
[-302.000000, 3.036873 ],
[342.000031, 2.199116  ],
[783.000061, 0.104720  ],
[491.000031, 0.000000  ],
[907.000061, 1.047198  ],
[151.000000, 1.884956  ],
[-140.000000, 2.932153 ],
[616.000000, 0.000000  ],
[1102.000000, 0.942478 ],
[247.000015, 1.989675  ],
[754.000061, 1.151917  ],
[465.000031, 1.256637  ],
[795.000061, 1.047198  ],
[1098.000000, 0.837758 ],
[916.000061, 0.942478  ],
[1021.500061, 0.942478 ],
[903.000061, 1.047198  ],
[791.000061, 1.047198  ],
[-305.000000, 3.036873 ],
[787.000061, 0.104720  ]
])

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

img = cv2.imread('1.jpg')

img1=img.copy()
for rho, theta in center1:
    (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
    cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)

img2=img.copy()
for rho, theta in center2:
    (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

img3=img.copy()
for rho, theta in center3:
    (x1,y1),(x2,y2) = polar2cartesian(rho,theta)
    cv2.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("v1",img1)
cv2.imshow("v2",img2)
cv2.imshow("v3",img3)
cv2.waitKey(0)