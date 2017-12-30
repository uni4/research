#http://www.tech-tech.xyz/archives/opencv_opening_closing.html
#オープニングとクロージング処理
import cv2
import numpy as np
import sys

filename = sys.argv[1]
#画像を読み込む
img = cv2.imread(filename)
#BGR画像をHSV画像に変換
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#HSVに分解
h_img, s_img, v_img = cv2.split(hsv)

#彩度のレイヤーを2値化してマスク画像を生成
_, mask_img = cv2.threshold(s_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#オープニングとクロージング
kernel = np.ones((2,2),np.uint8)
mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
kernel = np.ones((25,25),np.uint8)
mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)

#マスク画像を合成
result = cv2.merge(cv2.split(img) + [mask_img])

#「マスク部分を透明化した結果画像」とマスク画像を書き出す
cv2.imwrite(filename[:filename.rfind(".")] + "_result.png", result)
cv2.imwrite(filename[:filename.rfind(".")] + "_mask.png", mask_img)