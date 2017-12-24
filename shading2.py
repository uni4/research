#画像にガウシアンフィルターを適用させるプログラム
#https://algorithm.joho.info/programming/python/opencv-gaussian-filter-py/

import cv2
import numpy as np
    
#def main():
# 入力画像を読み込み
img = cv2.imread("org.jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
# 方法3
dst3 = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=1.3)

#画像を表示
cv2.imshow("GaussianBlue", dst3)
    
# 結果を出力
cv2.imwrite("output3.jpg", dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()
 
#if __name__ == "__main__":
#    main()