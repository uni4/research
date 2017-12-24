#画像の平滑化を行うプログラム(ローパスフィルタ)
#http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_filtering/py_filtering.html
#処理後色がおかしくなるが、これは画像がRGBなのに対して処理ではBGRで行なっているから

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

t1 = time.time()

img = cv2.imread('opencv.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.savefig("bokashi2.jpg")
plt.show()

t2 = time.time()
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")