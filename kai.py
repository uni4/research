import cv2
import numpy as np
import sys
import os.path

def main():
    # 入力画像の読み込み
    img = cv2.imread("%s"%param[1])
    height, width = img.shape[:2]
    name, ext = os.path.splitext(param[1])
    print(name)
    # 画像の中心座標
    oy, ox = int(height/2), int(width/2)
    i = 1

    for rn in range(-30, 30):

        theta = rn # 回転角
        scale = 1.0    # 回転角度・拡大率

        

        #画像の回転
        R = cv2.getRotationMatrix2D((oy, ox), theta, scale)    # 回転変換行列の算出
        dst = cv2.warpAffine(img, R, (width,height), flags=cv2.INTER_CUBIC)
        #dst = cv2.warpAffine(img, R, (height,width), flags=cv2.INTER_CUBIC)
        #dst[dst==[0,0,0]] = [255,255,255]
        dst = np.where(dst==[0,0,0], [255,255,255],dst)
        dst = np.where(dst==[0,0,0], [255,255,255],dst)

        # 結果を出力
        cv2.imwrite("..//pos/" + name + "_" + str(i) + ".jpg", dst)
        i += 1


if __name__ == "__main__":
    param = sys.argv
    main()