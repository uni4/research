from PIL import Image
import numpy as np
import sys
import cv2

kernel = np.ones((5, 5), np.uint8)

src = np.array(Image.open("skin.jpg").convert("L"))
height, width = src.shape[:2]

ret, src = cv2.threshold(src, 105, 255, cv2.THRESH_BINARY)
label = cv2.connectedComponentsWithStats(cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel))

# 2次元ままだと何かと処理しづらいので1次元に落とす
dst = label[1].reshape((-1))

# 背景を除いて1番大きいラベルのみを白にし、その他を黒とする
data = np.delete(label[2], 0, 0)
max_index = np.argsort(data[:,4])[::-1][0] + 1
for index in range(len(dst)):
    if dst[index] == max_index:
        dst[index] = 255;
    else:
        dst[index] = 0;

# 2次元に戻して画像として出力
dst = dst.reshape((height, width))
cv2.imwrite("kekka.png", dst)
cv2.imshow("image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
