#https://qiita.com/hitomatagi/items/2c3a2bfefe73ab5c86a4
#エッジ検出

import cv2

# 定数定義
ORG_WINDOW_NAME = "hand"
GRAY_WINDOW_NAME = "gray"
CANNY_WINDOW_NAME = "canny"

ORG_FILE_NAME = "hand.jpg"
GRAY_FILE_NAME = "gray.png"
CANNY_FILE_NAME = "canny.png"

#閾値
t = 220

# 元の画像を読み込む
org_img = cv2.imread(ORG_FILE_NAME, cv2.IMREAD_UNCHANGED)
# グレースケールに変換
gray_img = cv2.imread(ORG_FILE_NAME, cv2.IMREAD_GRAYSCALE)
#2値化      
ret,th2 = cv2.threshold(gray_img, t, 255, cv2.THRESH_BINARY)
# エッジ抽出
#canny_img = cv2.Canny(gray_img, 50, 110)
canny_img = cv2.Canny(th2, 50, 110)

#輪郭検出
#image, cnts, hierarchy = cv2.findContours(org_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#輪郭描画
#org_img = cv2.drawContours(im,contours,-1,(0,255,0),3)

# ウィンドウに表示
cv2.namedWindow(ORG_WINDOW_NAME)
cv2.namedWindow(GRAY_WINDOW_NAME)
cv2.namedWindow(CANNY_WINDOW_NAME)

cv2.imshow(ORG_WINDOW_NAME, org_img)
cv2.imshow(GRAY_WINDOW_NAME, gray_img)
cv2.imshow(CANNY_WINDOW_NAME, canny_img)

# ファイルに保存
cv2.imwrite(GRAY_FILE_NAME, gray_img)
cv2.imwrite(CANNY_FILE_NAME, canny_img)

# 終了処理
cv2.waitKey(0)
cv2.destroyAllWindows()