import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import Counter

#最大値と最小値を使った正規化
def min_max(x, axis=None):
	mi = min(x)
	ma = max(x)
	result = (x-mi)/(ma-mi)
	return result

#元データを平均0、標準偏差が1のものに変換する正規化
def zscore(x):
	xmean = x.mean()
	xstd  = np.std(x)

	zscore = (x-xmean)/xstd
	return zscore

def hist(img):
	#name, ext = os.path.splitext(param[1])
	# 複数色のチャンネルを分割して配列で取得
	# img_bgr[0] に青, img_bgr[1]に緑,img_bgr[2]に赤が入る。
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	img_bgr = cv2.split(hsv)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	b = clahe.apply(img_bgr[0])
	g = clahe.apply(img_bgr[1])
	r = clahe.apply(img_bgr[2])
	# b = cv2.equalizeHist(img_bgr[0])
	# g = cv2.equalizeHist(img_bgr[1])
	# r = cv2.equalizeHist(img_bgr[2])

	img_hist_seiki = cv2.merge((b,g,r))
	img_hist = cv2.merge((img_bgr[0],img_bgr[1],r))
	cv2.imwrite("hist_moto.jpg",hsv)
	cv2.imwrite("hist.jpg",img_hist)
	cv2.imwrite("hist_seiki.jpg",img_hist_seiki)

	return img_hist



	# 表示
	"""
	cv2.imshow("Show Image", img_hist)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
def hada(im):
	#im = cv2.imread("%s"%param[1])
	im = photo_cut(im)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	#im = hada_range(im)
	img_bgr = cv2.split(im)
	threshold = 0.1

	h = img_bgr[0].flatten()
	s = img_bgr[1].flatten()
	v = img_bgr[2].flatten()
	hsv_min = [2,2]
	hsv_max = [-1,-1]
	hsv_threshold = [hsv_min,hsv_max]
	s_count =np.array([])
	v_count = np.array([])
	s_sum = np.array([])
	v_sum = np.array([])

	max_sv = []
	index = 0
	for i in range(0,257):
		s_sum = s_sum + np.sum(s==i)
		v_sum = v_sum + np.sum(v==i)
		if i%8==0 and i >16:
			s_count = np.append(s_count,s_sum)
			v_count = np.append(v_count,v_sum)
			s_sum = 0
			v_sum = 0
			#s_count = np.append(s_count,np.array([np.sum(s==i)]))
			#v_count = np.append(v_count,np.array([np.sum(v==i)]))
			index +=1
	s_max = s_count.argmax() * 8 +16
	v_max = v_count.argmax() * 8 +16
	"""
	#肌色の閾値を決める
	まず一番値が集中しているところを基準値とする。その前後の値の数を
	
	#HSVの各要素数を求める
	color_number = len(h)
	a = 0
	flag = True
	s_sort = np.argsort(x)[::-1][i]
	s_max = s_count.argmax() * 8 +16
	v_max = v_count.argmax() * 8 +16
	
	s_seiki = min_max(s_count)
	v_seiki = min_max(v_count)
	bins_range = range(0, len(s_seiki), 1)
	xtics_range = range(0, len(s_seiki), 1)
	"""

	if s_max-40 < 0:
		max_sv.append(16)
	else:
		max_sv.append(s_max-40)
	if s_max+40 > 255:
		max_sv.append(255)
	else:
		max_sv.append(s_max+40)
	if v_max-40 < 0:
		max_sv.append(0)
	else:
		max_sv.append(v_max-40)
	if v_max+40 > 255:
		max_sv.append(255)
	else:
		max_sv.append(v_max+40)

	#max_sv = []
	print(max_sv)

	"""
	for i in range(0,257):
		if hsv_max[0]<s_count[i] and s_count[i]>threshold:
			hsv_max[0] =  i
		if hsv_max[1]<v_count[i] and v_count[i]>threshold:
			hsv_max[1] =  i
		if hsv_min[0]>s_count[i] and s_count[i]>threshold:
			hsv_min[0] = i
		if hsv_min[1]>v_count[i] and v_count[i]>threshold:
			hsv_min[1] =  i
	
	
			
	bins_range = range(0, 257, 8)
	xtics_range = range(0, 257, 32)

	plt.hist((h, s, v), bins=bins_range,
			 color=['r', 'g', 'b'], label=['Red', 'Green', 'Blue'])
	plt.legend(loc=2)

	plt.grid(True)

	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 256, 0, ymax])
	plt.xticks(xtics_range)

	plt.savefig("histogram_single2.jpg")
	"""
	
	#return hsv_threshold
	return max_sv


def hada_range(im):
	im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	height,width = im.shape[:2]
	img_bgr = cv2.split(im)

	h = img_bgr[0].flatten()
	s = img_bgr[1].flatten()
	v = img_bgr[2].flatten()

	ran_image = []
	h_max = 0

	height_split = 4
	width_split = 4
	new_img_height = int(height / height_split)
	new_img_width = int(width / width_split)

	for h in range(height_split):
		height_start = h * new_img_height
		height_end = height_start + new_img_height
		for w in range(width_split):
			width_start = w * new_img_width
			width_end = width_start + new_img_width
			clp = im[height_start:height_end, width_start:width_end]
			clp_bgr = cv2.split(clp)
			h = clp_bgr[0].flatten()
			s = clp_bgr[1].flatten()
			v = clp_bgr[2].flatten()
			num =  len(np.where((h>0) & (h<30))[0])
			if h_max < num:
				h_max = num
				ran_image = clp
	return ran_image

def photo_cut(img):
	height,width = img.shape[:2]
	xmin = int(width/2 - width/4)
	xmax = int(width/2 + width/4)
	ymin = int(height/2 - height/4)
	ymax = int(height/2 + height/4)
	img_copy = img[ymin:ymax, xmin:xmax]
	cv2.imwrite("cut.jpg",img_copy)
	return img_copy

#肌色領域のみを抽出
def mask(img):
	height, width = img.shape[:2]
	
	# フレームをHSVに変換
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	htd = hada(img)
	# 取得する色の範囲を指定する
	#lower = np.array([0, 0, 190])
	#upper = np.array([30, 100, 255])
	lower = np.array([0, htd[1][0], htd[1][1]])
	upper = np.array([30, htd[0][0], htd[0][1]])

	# 指定した色に基づいたマスク画像の生成
	img_mask = cv2.inRange(hsv, lower, upper)

	#フレーム画像とマスク画像の共通の領域を抽出する。
	img_color = cv2.bitwise_and(img, img, mask=img_mask)
	cv2.imwrite("out.jpg",img_color)

	return img_color

def main():
	# 入力画像を読み込み
	img = cv2.imread("%s"%param[1])
	#hist(img)
	#hada(img)
	#photo_cut(img)
	mask(img)


	
if __name__ == "__main__":
	param = sys.argv
	main()