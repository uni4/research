#画像から肌色領域のみを抽出するプログラム
#https://www.blog.umentu.work/python3-opencv3%E3%81%A7%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%9F%E8%89%B2%E3%81%AE%E3%81%BF%E3%82%92%E6%8A%BD%E5%87%BA%E3%81%97%E3%81%A6%E8%A1%A8%E7%A4%BA%E3%81%99%E3%82%8B%E3%80%90%E5%8B%95%E7%94%BB/

import cv2
import numpy as np
import sys

def mask(param):
    # 画像を取得
    img = cv2.imread("%s"%param[1])

    # フレームをHSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 取得する色の範囲を指定する
    lower = np.array([0, 30, 160])
    upper = np.array([30, 150, 255])

    # 指定した色に基づいたマスク画像の生成
    img_mask = cv2.inRange(hsv, lower, upper)

    #フレーム画像とマスク画像の共通の領域を抽出する。
    img_color = cv2.bitwise_and(img, img, mask=img_mask)
    #img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)

    cv2.imwrite("color.jpg", img_color)

    return img_color

def labelling(im):
    height, width = im.shape[:2]
    kernel = np.ones((5,5),np.uint8)

    # グレースケール変換
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 2値化
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # ラベリング処理
    label = cv2.connectedComponentsWithStats(gray)

    data = np.delete(label[2], 0, 0)
    max_index = np.argsort(data[:,4])[::-1][0] + 1
    dst = label[1].reshape((-1))

    for index in range(len(dst)):
        if dst[index] == max_index:
            dst[index] = 255;
        else:
            dst[index] = 0;

    dst = dst.reshape((height, width))
    cv2.imwrite("kekka_totyu.jpg",dst)

    im_re = cv2.imread("kekka_totyu.jpg")

    # グレースケール変換
    gray2 = cv2.cvtColor(im_re, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    image, contours, hierarchy = cv2.findContours(gray2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im_con = cv2.drawContours(im_re, contours, 0, (255,255,255), -1,)

    cv2.imwrite("kekka.jpg",im_con)
    opening = cv2.morphologyEx(im_con, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(im_con, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("opening.jpg", opening)
    cv2.imwrite("closing.jpg", closing)

if __name__ == '__main__':
    param = sys.argv
    labelling(mask(param))