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
	im = photo_cut(im)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	#im = hada_range(im)
	img_bgr = cv2.split(im)

	h = img_bgr[0].flatten()
	s = img_bgr[1].flatten()
	v = img_bgr[2].flatten()
	max_in = np.array([])

	max_sv = []

	bins_range = range(0, 257, 8)
	s_hist = np.histogram((s), bins=bins_range)
	v_hist = np.histogram((v), bins=bins_range)
	#print("s値",s_hist[0])
	#print("v値",v_hist[0])


	flag = True
	index = 0
	max_index = 0
	s_sum = 0
	s_range = np.array([])
	total_tresh = sum(s_hist[0][2:len(s_hist[0])]) *0.45
	total_tresh2 = sum(v_hist[0][2:len(v_hist[0])]) *0.45
	while flag == True:	
		max_index = np.argsort(s_hist[0])[::-1][index]
		#print("max_index",s_hist[0][max_index])
		s1_index = max_index -1
		s2_index = max_index +1
		tresh = s_hist[0][max_index] * 0.25
		#print("tesh",tresh)
		s_sum = s_sum + s_hist[0][max_index]
		s_range = np.append(s_range,max_index)
		while s1_index > 0 and s_hist[0][s1_index] > tresh and max_index > 2:
			s_sum = s_sum + s_hist[0][s1_index]
			s_range = np.append(s_range,s1_index)
			s1_index -= 1 
		while s2_index < len(s_hist[0]) and s_hist[0][s2_index] > tresh and max_index > 2:
			s_sum = s_sum + s_hist[0][s2_index]
			s_range = np.append(s_range,s2_index)
			s2_index += 1
		if s_sum >  total_tresh and max_index > 2:
			#print("sの最終最大値",s_sum)
			flag = False
			max_in = np.append(max_in,s_hist[1][max_index])
		else:
			#print("sの最大値",s_sum)
			index+=1
			s_range = np.array([])
			s_sum = 0

	#print("")
	flag = True
	index = 0
	max_index = 0
	v_range = np.array([])	
	v_sum = 0			
	while flag == True:	
		max_index = np.argsort(v_hist[0])[::-1][index]
		#print("v_max_index",max_index*8 +24)
		v1_index = max_index -1
		v2_index = max_index +1
		tresh = v_hist[0][max_index] * 0.25
		#print("max_index",v_hist[0][max_index])
		#print("閾値",tresh)
		v_sum = v_sum + v[max_index]
		v_range = np.append(v_range,max_index)
		while v1_index > 0 and v_hist[0][v1_index] > tresh and max_index > 2:
			v_sum = v_sum + v_hist[0][v1_index]
			v_range = np.append(v_range,v1_index)
			v1_index -= 1 
		while v2_index < len(v_hist[0]) and v_hist[0][v2_index] > tresh and max_index > 2:
			v_sum = v_sum + v_hist[0][v2_index]
			v_range = np.append(v_range,v2_index)
			v2_index += 1
		if v_sum >  total_tresh2 and max_index > 2:
			flag = False
			#print("vの最終最大値",max_index*8 +24)
			max_in = np.append(max_in,v_hist[1][max_index])
		else:
			#print("vの最大値",v_sum)
			index+=1
			v_range = np.array([])
			v_sum = 0

	s_range = np.sort(s_range)
	v_range = np.sort(v_range)

	max_sv.extend([s_range[0]*8 -8,s_range[-1]*8 +8])
	max_sv.extend([v_range[0]*8 -8,v_range[-1]*8 +8])
	max_sv.extend(max_in)
	if max_sv[0] < 8:
		max_sv[0] = 8

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
	#cv2.imwrite("cut.jpg",img_copy)
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
	#cv2.imwrite("out.jpg",img_color)

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