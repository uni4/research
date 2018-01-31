import cv2
import numpy as np
import sys
import os.path

def threshold(src, ksize, c):
	
	# 畳み込み演算をしない領域の幅
	d = int((ksize-1)/2)
	h, w = src.shape[0], src.shape[1]

	# 出力画像用の配列（要素は全て255）
	dst = np.empty((h,w))
	dst.fill(255)

	n = ksize**2

	for y in range(0, h):
		for x in range(0, w):
			# 近傍の画素値の平均から閾値を求める
			t = np.sum(src[y-d:y+d+1, x-d:x+d+1]) / n
			# 求めた閾値で二値化処理
			if(src[y][x] < t - c): dst[y][x] = 0
			else: dst[y][x] = 255

	return dst


def shimon():
	# 入力画像を読み込み
	img = cv2.imread("shimon/" + "%s"%param[1])
	name, ext = os.path.splitext(param[1])
	if os.path.exists("/Users/dennomaaya/Desktop/py/shimon/" + str(name)) == False:
		os.mkdir("/Users/dennomaaya/Desktop/py/shimon/" + str(name))

	# グレースケール変換
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	for ksize in range(5, 33, 2):
		# 方法2       
		dst2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,ksize,2)
		#dst2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,ksize,2)
		#dst2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,ksize,2)黒い
		#dst2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,ksize,2)黒くなる
		#dst2 = threshold(gray, ksize, c=2)
		#dst2 = cv2.resize(dst2, (504, 480), interpolation=cv2.INTER_LINEAR)

		# 結果を出力
		cv2.imwrite("shimon/" + name + "/" + name + "_shimon" + str(ksize) + ".tif", dst2)

def recog_shimon():
	img = cv2.imread("shimon/" + "%s"%param[1])
	name, ext = os.path.splitext(param[1])
	if os.path.exists("/Users/dennomaaya/Desktop/py/shimon/" + str(name)) == False:
		os.mkdir("/Users/dennomaaya/Desktop/py/shimon/" + str(name))

	for ksize in range(5, 23, 2):
		# グレースケール変換
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		#画像を左右反転させる
		imgaxis_y = cv2.flip(gray, 1)
		imgtone_flip = invgray = cv2.bitwise_not(imgaxis_y)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ksize,ksize))
		cl1 = clahe.apply(imgaxis_y)
		dst2 = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,ksize,1)
		cv2.imwrite("shimon/" + name + "/" + name + "_shimon" + str(ksize) + ".bmp", dst2)



def main():
	shimon()
	#recog_shimon()

if __name__ == "__main__":
	param = sys.argv
	main()