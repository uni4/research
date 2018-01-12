#研究用プログラム
#指紋を消すプログラム
import cv2
import numpy as np
import sys
from scipy.spatial import distance
import time
import print as pt

#肌色領域のみを抽出
def mask(param):
	# 画像を取得
	img = cv2.imread("%s"%param[1])

	# フレームをHSVに変換
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# 取得する色の範囲を指定する
	lower = np.array([0, 30, 130])
	upper = np.array([30, 150, 255])

	# 指定した色に基づいたマスク画像の生成
	img_mask = cv2.inRange(hsv, lower, upper)

	#フレーム画像とマスク画像の共通の領域を抽出する。
	img_color = cv2.bitwise_and(img, img, mask=img_mask)
	#img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)

	cv2.imwrite("color.jpg", img_color)
	print("肌色領域抽出")

	return img_color

#最大面積を抽出することでノイズを除去
def labelling(im):
	height, width = im.shape[:2]
	kernel = np.ones((10,10),np.uint8)

	# グレースケール変換
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	label = cv2.connectedComponentsWithStats(gray)

	data = np.delete(label[2], 0, 0)
	max_index = np.argsort(data[:,4])[::-1][0] + 1
	dst = label[1].reshape((-1))

	print("最大面積領域の抽出開始")
		
	dst = np.array([255 if dst[index] == max_index else 0 for index in range(len(dst))])

	dst = dst.reshape((height, width))
	cv2.imwrite("kekka_totyu.jpg",dst)
	print("最大面積領域を抽出")
	im_re = cv2.imread("kekka_totyu.jpg")

	# グレースケール変換
	gray2 = cv2.cvtColor(im_re, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	#第3結果を0にすれば一番外側の輪郭のみを描画するため中身を全て白にできる
	image, contours, hierarchy = cv2.findContours(gray2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im_con = cv2.drawContours(im_re, contours, 0, (255,255,255), -1,)

	cv2.imwrite("kekka.jpg",im_con)
	closing = cv2.morphologyEx(im_con, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("closing10.jpg", closing)
	print("クロージング処理終了")

	return closing

#指先を抽出する処理
def yubisaki(im):

	height,width = im.shape[:2]
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv2.rectangle(gray, (height-200, height-1), (width-1, height-1), (0, 0, 0), -1)

	label = cv2.connectedComponentsWithStats(gray)
	# ブロブ情報を項目別に抽出
	n = label[0] - 1
	data = np.delete(label[2], 0, 0)
	center = np.delete(label[3], 0, 0)
	max_index = np.argsort(data[:,4])[::-1][0] + 1

	x_center = int(center[max_index -1][0])
	y_center = int(center[max_index -1][1])
	
	dst = label[1]
	list = []
	list2 = []
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
				dis = distance.euclidean(a,c)
				list2.append(dis - ave)
				if (dis - ave) < 800:
					im[y][x] = (0,0,0)

	print("1c")
	ave2 = np.average(list2)
	su = np.sum(list2)
	cv2.imwrite("saki.jpg", im)

	return im

def main():
	print(param)
	t1 = time.time() 

	gazou = labelling(mask(param))
	gazou2 = yubisaki(gazou)
	pt.finger_edit(param, gazou2)

	t2 = time.time()
	elapsed_time = t2-t1
	print(f"経過時間：{elapsed_time}")

if __name__ == '__main__':
	param = sys.argv
	main()