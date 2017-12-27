#https://algorithm.joho.info/programming/python/blob-max-moment/
#http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
import cv2
import numpy as np

def main():
    # 入力画像の取得
    im = cv2.imread("src.png")
    height = im.shape[0]
    width = im.shape[1]

    #黒色の画像を作成
    imageArray = np.zeros((height, width, 3), np.uint8)
    
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

    # ブロブ面積最大のインデックス
    max_index = np.argmax(data[:,4])
    max_area = 0
    print("最大面積", max_index)

    #輪郭情報の取得及び描画
    image, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[4]
    #cnt = [i for i in hierarchy[0,-2,0] if cv2.contourArea(contours[i]) > max_area]
    im_con = cv2.drawContours(im, contours, -1, (255,0,0), 3)

    # 面積最大ブロブの各種情報を表示
    print("外接矩形の左上x座標", data[:,0][max_index])
    print("外接矩形の左上y座標", data[:,1][max_index])
    print("外接矩形の幅", data[:,2][max_index])
    print("外接矩形の高さ", data[:,3][max_index])
    print("面積", data[:,4][max_index])
    print("中心座標:\n",center[max_index])
    print("ラベル", data)
    print("ラベル数", n)
    print("輪郭", hierarchy[0,-2,0])
    print("データ", center)

    h = int(center[max_index][0])
    w = int(center[max_index][1])
    img2 = cv2.circle(im_con,(h,w), 1, (0,0,255), 3)

    #resized_img = cv2.resize(img2,(333,444))
    cv2.imshow("SHOW COLOR IMAGE", img2)
    cv2.imshow("label_image", label[1][:])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()