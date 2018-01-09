#http://www.cellstat.net/pycvmoment/
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

param = sys.argv
img = cv2.imread("%s"%param[1],cv2.IMREAD_GRAYSCALE)

mu = cv2.moments(img, False)
x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])

cv2.circle(img, (x,y), 4, 100, 255, -1)
plt.imshow(img)
plt.colorbar()
plt.show()

print(x,y)