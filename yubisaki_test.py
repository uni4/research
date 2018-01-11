#指先を検出するプログラム
import cv2
import numpy as np
import sys
from scipy.spatial import distance
import time

def main():
    # 入力画像の取得
    t1 = time.time() 
    im = cv2.imread("%s"%param[1])
    height,width = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.rectangle(gray, (0, height-200), (width-1, height-1), (0, 0, 0), -1)

    label = cv2.connectedComponentsWithStats(gray)
    # ブロブ情報を項目別に抽出
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    max_index = np.argsort(data[:,4])[::-1][0] + 1
    #print("最大面積のラベル番号", max_index)
    x_center = int(center[max_index -1][0])
    y_center = int(center[max_index -1][1])
    
    dst = label[1]
    list = []
    list2 = []
    #a = np.array(center[max_index -1])
    a = np.array([y_center,x_center])
    print("1a")
    
    for y in range(height):
        for x in range(width):
            if dst[y][x] == max_index:
                b = np.array([y,x])
                list.append(distance.euclidean(a,b))

    #ave = np.mean(list)
    ave = np.average(list)
    print("1b")

    for y in range(height):
        for x in range(width):
            if dst[y][x] == max_index:
                c = np.array([y,x])
                #dis = np.linalg.norm(a-c)
                dis = distance.euclidean(a,c)
                list2.append(dis - ave)
                #print(dis - ave)
                if (dis - ave) < 1000:
                    im[y][x] = (255,255,0)

    print("1c")
    ave2 = np.average(list2)
    su = np.sum(list2)
    
    print("中心との距離の平均",ave)
    print("偏差の平均",ave2)
    print("偏差の和",su)
    
    """
    # ラベルの個数nだけ色を用意
    print("ブロブの個数:", n)
    print("各ブロブの外接矩形の左上x座標", data[:,0])
    print("各ブロブの外接矩形の左上y座標", data[:,1])
    print("各ブロブの外接矩形の幅", data[:,2])
    print("各ブロブの外接矩形の高さ", data[:,3])
    print("各ブロブの面積", data[:,4])
    print("中心座標:\n",center[max_index - 1])
    """

    cv2.circle(im,(x_center,y_center), 20,(0,0,255),-1)
    cv2.imwrite("yubisaki.jpg", im)
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"経過時間：{elapsed_time}")


if __name__ == '__main__':
    param = sys.argv
    main()