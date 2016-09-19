import numpy as np
import cv2
from matplotlib import pyplot as plt

z = [
0.942478,
3.036873,
2.199116 ,
0.104720 ,
0.000000 ,
1.047198 ,
1.884956 ,
2.932153,
0.000000 ,
0.942478,
1.989675 ,
1.151917 ,
1.256637 ,
1.047198 ,
0.837758,
0.942478 ,
0.942478,
1.047198 ,
1.047198 ,
3.036873,
0.104720
]

for i in range(0,len(z)):
    if z[i]<0.5:
        z[i]+= np.pi

z = np.float32(z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compactness,labels,centers = cv2.kmeans(z,3,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

A = z[labels.ravel()==0]
B = z[labels.ravel()==1]
C = z[labels.ravel()==2]

print z.shape
print A.shape
