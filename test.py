import cv2
import numpy as np
import sys
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt


#最大値と最小値を使った正規化
def min_max(x, axis=None):
	mi = min(x)
	ma = max(x)
	result = (x-mi)/(ma-mi)
	return result

def seiki():
	list = np.array([[1, 2, 3],
					   [4, 5, 6],
					   [7, 8, 9]])
	ave = np.average(list)
	hensa = [list - ave]
	su = np.sum(hensa)
	print("平均",ave)
	print("偏差",hensa)
	print("偏差の合計",su)

def kyori2():
	list = []
	index = 1
	dst = np.random.randint(1, 4, (100, 100))
	dst2 = np.ones((100,100))
	num1 =  len(np.where(dst2==1)[0])
	print("処理前の1の要素数",num1)
	height,width = dst.shape[:2]
	a = [50,50]

	x=np.arange(width).reshape([1,width])
	y=np.arange(height).reshape([height,1])
	euclidean=((x-a[0])**2+(y-a[1])**2)**0.5
	euclidean = euclidean.reshape((-1))
	euclidean = min_max(euclidean)
	euclidean = euclidean.reshape((height, width))
	avg=np.average(euclidean[dst==index]) 
	print(avg)
	euclidean[(dst==index)*(euclidean<avg)]=0
	dst2[euclidean==0] = 0

	num2 =  len(np.where(dst2==1)[0])
	print("処理後の1の要素数",num2)
	num3 =  len(np.where(dst2==0)[0])
	print("処理後の0の要素数",num3)

def kyori3():
	list = []
	index = 1
	dst = np.random.randint(1, 4, (100, 100))
	dst2 = np.ones((100,100))
	num1 =  len(np.where(dst2==1)[0])
	print("処理前の1の要素数",num1)
	height,width = dst2.shape[:2]
	a = [50,50]

	for y in range(height):
		for x in range(width):
			if dst[y][x] == index:
				b = np.array([y,x])
				list.append(distance.euclidean(a,b))
	#速度向上のための案
	#list = np.array([distance.euclidean(a,dst[dst==index])])
	list = min_max(list)
	ave = np.average(list)
	print(ave)


	i = 0
	for y in range(height):
		for x in range(width):
			if dst[y][x] == index:
				i += 1
				if list[i-1] < ave:
					dst2[y][x] = 0

	num2 =  len(np.where(dst2==1)[0])
	print("処理後の1の要素数",num2)
	num3 =  len(np.where(dst2==0)[0])
	print("処理後の0の要素数",num3)


def histgram():

	#im = Image.open('hist_test.jpg')
	im = cv2.imread("%s"%param[1])
	#im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
	img_bgr = cv2.split(im)

	"""
	r = np.array(im)[:, :, 0].flatten()
	g = np.array(im)[:, :, 1].flatten()
	b = np.array(im)[:, :, 2].flatten()
	lower = np.array([0, 30, 130])
	upper = np.array([30, 150, 255])
	lower_hcc = np.array([100, 80, 50])
	upper_hcc = np.array([235, 200, 240])
	"""
	#肌色の範囲
	lower_hcc = np.array([130, 80, 80])
	upper_hcc = np.array([240, 160, 140])
	img_mask_hcc = cv2.inRange(im, lower_hcc, upper_hcc)
	#肌色の範囲で赤っぽい所
	lower_red = np.array([100, 130, 90])
	upper_red = np.array([130, 160, 140])
	img_mask_red = cv2.inRange(im, lower_red, upper_red)
	img_mask_hcc[img_mask_red==255] = 255
	#肌色で黒っぽい所
	lower_black = np.array([64, 150, 90])
	upper_black = np.array([90, 170, 120])
	img_mask_black = cv2.inRange(im, lower_black, upper_black)
	img_mask_hcc[img_mask_black==255] = 255

	# 指定した色に基づいたマスク画像の生成
	#img_mask_hcc = cv2.inRange(im, lower_hcc, upper_hcc)

	#フレーム画像とマスク画像の共通の領域を抽出する。
	img_color = cv2.bitwise_and(im, im, mask=img_mask_hcc)
	cv2.imwrite("img_hcc.jpg",img_mask_hcc)

	h = img_bgr[0].flatten()
	s = img_bgr[1].flatten()
	v = img_bgr[2].flatten()

	bins_range = range(0, 257, 8)
	xtics_range = range(0, 257, 32)

	plt.hist((h, s, v), bins=bins_range,
	         color=['r', 'g', 'b'], label=['Red', 'Green', 'Blue'])
	plt.legend(loc=2)

	plt.grid(True)

	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 256, 0, ymax])
	plt.xticks(xtics_range)

	plt.savefig("histogram_single.jpg")


def main():
	histgram()




if __name__ == '__main__':
	param = sys.argv
	main()