#指先を検出するプログラム
import cv2
import numpy as np
import sys
from scipy.spatial import distance
import time

def main():
    c_kernel = np.ones((3,3),np.uint8)
    img_cc = cv2.imread("%s"%param[1])
    height,width = img_cc.shape[:2]
    img_cc = cv2.morphologyEx(img_cc, cv2.MORPH_OPEN, c_kernel)
    img_cc = cv2.morphologyEx(img_cc, cv2.MORPH_OPEN, c_kernel)
    #img_cc = cv2.morphologyEx(img_cc, cv2.MORPH_CLOSE, c_kernel)
    #img_cc = cv2.morphologyEx(img_cc, cv2.MORPH_CLOSE, c_kernel)
    cv2.imwrite("molpho.jpg",img_cc)


if __name__ == '__main__':
    param = sys.argv
    main()