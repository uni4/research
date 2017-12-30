import cv2
import numpy as np

def main():
    # 入力画像の取得
    im = cv2.imread("skin.jpg")
    height = im.shape[0]
    width = im.shape[1]

    # グレースケール変換
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 2値化
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # ラベリング処理
    label = cv2.connectedComponentsWithStats(gray)

    # ブロブ情報を項目別に抽出
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    max_index = np.argsort(data[:,4])[::-1][0] + 1
    print("最大面積のラベル番号", max_index)

    dst = label[1].reshape((-1))

    #ある特定の数値の要素数を数える
    num = len(np.where(dst == 1)[0])
    print("要素数", num)

    for index in range(len(dst)):
        if dst[index] == max_index:
            dst[index] = 255;
        else:
            dst[index] = 0;

    num1 = len(np.where(dst == 0)[0])
    print("要素数", num1)

    dst = dst.reshape((height, width))

    # ラベルの個数nだけ色を用意
    print("ブロブの個数:", n)
    print("各ブロブの外接矩形の左上x座標", data[:,0])
    print("各ブロブの外接矩形の左上y座標", data[:,1])
    print("各ブロブの外接矩形の幅", data[:,2])
    print("各ブロブの外接矩形の高さ", data[:,3])
    print("各ブロブの面積", data[:,4])
    #print("各ブロブの中心座標:\n",center)

    #cv2.imshow("gazou", dst)
    cv2.imwrite("kekka_test.jpg", dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()