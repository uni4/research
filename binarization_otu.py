#https://algorithm.joho.info/programming/python/opencv-otsu-thresholding-py/#Python3OpenCV3
#大津の2値化

import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("aaa.jpg")

    # フレームをHSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    # 取得する色の範囲を指定する
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
 
    # 指定した色に基づいたマスク画像の生成
    img_mask = cv2.inRange(hsv, lower, upper)
 
    #フレーム画像とマスク画像の共通の領域を抽出する。
    img_color = cv2.bitwise_and(img, img, mask=img_mask)

    # グレースケール変換
    gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    
    # 方法2 （OpenCVで実装）      
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)    

    # 結果を出力
    cv2.imwrite("th2.jpg", th2)
    cv2.imwrite("th1.jpg", img_color)


if __name__ == "__main__":
    main()
