#画像から肌色領域のみを抽出するプログラム
#https://www.blog.umentu.work/python3-opencv3%E3%81%A7%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%9F%E8%89%B2%E3%81%AE%E3%81%BF%E3%82%92%E6%8A%BD%E5%87%BA%E3%81%97%E3%81%A6%E8%A1%A8%E7%A4%BA%E3%81%99%E3%82%8B%E3%80%90%E5%8B%95%E7%94%BB/

import cv2
import numpy as np
  
# 画像を取得
img = cv2.imread("aa.jpg")
 
# フレームをHSVに変換
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
# 取得する色の範囲を指定する
lower = np.array([0, 30, 60])
upper = np.array([20, 150, 255])
 
# 指定した色に基づいたマスク画像の生成
img_mask = cv2.inRange(hsv, lower, upper)
 
#フレーム画像とマスク画像の共通の領域を抽出する。
img_color = cv2.bitwise_and(img, img, mask=img_mask)
 
cv2.imshow("SHOW COLOR IMAGE", img_color)


# ファイルに保存
cv2.imwrite("skin.jpg", img_color)
 
    # qを押したら終了
    #k = cv2.waitKey(1)
    #if k == ord('q'):
    #    break

    # 終了処理
cv2.waitKey(0)
cv2.destroyAllWindows()