#画像をBGRからHSVに変換する
#https://algorithm.joho.info/programming/python/opencv-rgb-to-hsv-color-space/

import numpy as np
import cv2

img = cv2.imread('hand.jpg')

# BGR空間から HSV空間に変換
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 結果を出力
cv2.imwrite("hsv.jpg", hsv)