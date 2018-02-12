#研究用プログラム
#指紋を消すプログラム
import cv2
import numpy as np
import sys
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
import os.path
import fingerprint_edit as pt
import hist
import datetime
from PIL import Image, ImageDraw, ImageFont
sys.path.append("create/study-master/objectdetection/models/research/object_detection/")
import recog_hand as recog

#ベクトルの正規化
def normalize(v, axis=-1, order=2):
	l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
	l2[l2==0] = 1
	return v/l2

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

 #画像の暗部を持ち上げる
 #http://peaceandhilightandpython.hatenablog.com/entry/2016/02/05/004445
def lookup(img_src,name,index):
	gamma = 1.8
	lookUpTable = np.zeros((256, 1), dtype = 'uint8')

	for i in range(256):
		lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

	img_gamma = cv2.LUT(img_src, lookUpTable)
	cv2.imwrite("work/" + str(name) + "/" + str(name) + str(index) +"_lookup.jpg", img_gamma)
	return img_gamma

#肌色領域のみを抽出
def mask(img,name,index):
	print(str(index) + "回目肌色領域抽出開始")
	height, width = img.shape[:2]
	
	# フレームをHSVに変換
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#Y : 198.562	Cb : -29.112	Cr : 30.252	
	#hcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	# 取得する色の範囲を指定する
	#lower_hcc = np.array([100, 64, 50])
	#upper_hcc = np.array([235, 160, 255])

	# 取得する色の範囲を指定する
	#lower = np.array([0, 30, 130])
	#upper = np.array([30, 150, 255])

	htd = hist.hada(img)
	lower = np.array([0, htd[0], htd[2]])
	upper = np.array([30, htd[1], htd[3]])
	
	
	height,width = img.shape[:2]
	xmin = int(width/2 - width/4)
	xmax = int(width/2 + width/4)
	ymin = int(height/2 - height/4)
	ymax = int(height/2 + height/4)
	img_copy = hsv[ymin:ymax, xmin:xmax]
	
	img_bgr = cv2.split(img_copy)
	h = img_bgr[0].flatten()
	s = img_bgr[1].flatten()
	v = img_bgr[2].flatten()
	plt.figure()
	bins_range = range(0, 257, 8)
	xtics_range = range(0, 257, 32)
	plt.hist((h, s, v), bins=bins_range,
			 color=['r', 'g', 'b'], label=['Hue', 'Saturation', 'Value'])
	plt.legend(loc=2)
	plt.grid(True)
	#plt.title(htd)
	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 256, 0, ymax])
	plt.xticks(xtics_range)
	img_copy = cv2.cvtColor(img_copy, cv2.COLOR_HSV2BGR)
	cv2.imwrite("work/" + str(name) + "/" + str(name) + str(index) +"_histrange.jpg", img_copy)
	plt.savefig("work/" + str(name) + "/" + str(name) + str(index) +"_hist.jpg")
	plt.figure()


	#元の手の範囲のヒストグラムの作成
	img_bgr = cv2.split(hsv)
	h = img_bgr[0].flatten()
	s = img_bgr[1].flatten()
	v = img_bgr[2].flatten()
	bins_range = range(0, 257, 8)
	xtics_range = range(0, 257, 32)
	plt.hist((h, s, v), bins=bins_range,
			 color=['r', 'g', 'b'], label=['Hue', 'Saturation', 'Value'])
	plt.legend(loc=2)
	plt.grid(True)
	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 256, 0, ymax])
	plt.xticks(xtics_range)
	plt.savefig("work/" + str(name) + "/" + str(name) + str(index) +"_histmoto.jpg")
	plt.figure()

	# 指定した色に基づいたマスク画像の生成
	img_mask = cv2.inRange(hsv, lower, upper)

	#フレーム画像とマスク画像の共通の領域を抽出する。
	img_color = cv2.bitwise_and(img, img, mask=img_mask)

	cv2.imwrite("work/" + str(name) + "/" + str(name) + str(index) +"_color.jpg", img_color)
	print(str(index) + "回目肌色領域抽出終了")

	return img_color

#最大面積を抽出することでノイズを除去
def labelling(im,name,index):
	print(str(index) + "回目最大面積領域の抽出開始")
	height, width = im.shape[:2]
	kernel = np.ones((10,10),np.uint8)

	# グレースケール変換
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	#gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("work/" + str(name) + "/" + str(name) + "_" + str(index) +"_gray.jpg",gray)


	"""
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	#第3結果を0にすれば一番外側の輪郭のみを描画するため中身を全て白にできる
	image, contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	gray = cv2.drawContours(gray, contours, -1, (255,255,255), -1,)
	#cv2.fillPoly(gray, pts =[contours], color=(0,0,255))
	cv2.imwrite("work/" + str(name) + "/" + str(name) + "_" + str(index) +"_gray2.jpg",gray)
	"""



	label = cv2.connectedComponentsWithStats(gray)

	data = np.delete(label[2], 0, 0)
	max_index = np.argsort(data[:,4])[::-1][0] + 1
	dst = label[1].reshape((-1))

	#最大面積のラベル番号の部分は255にして、それ以外を0にする
	dst[dst==max_index] = 255
	dst[dst!=255] = 0

	dst = dst.reshape((height, width))
	cv2.imwrite("work/" + str(name) + "/" + str(name) + "_" + str(index) +"_totyu.jpg",dst)
	print(str(index) + "回目最大面積領域を抽出終了")
	im_re = cv2.imread("work/" + str(name) + "/" + str(name) + "_" + str(index) +"_totyu.jpg")

	print(str(index) + "回目クロージング処理開始")
	# グレースケール変換
	gray2 = cv2.cvtColor(im_re, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.morphologyEx(gray2, cv2.MORPH_CLOSE, kernel)
	#gray2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	gray2 = cv2.threshold(gray2,10,255,cv2.THRESH_BINARY)[1]


	#第3結果を0にすれば一番外側の輪郭のみを描画するため中身を全て白にできる
	image, contours, hierarchy = cv2.findContours(gray2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	im_con = cv2.drawContours(im_re, contours, -1, (255,255,255), -1,)

	cv2.imwrite("work/" + str(name) + "/" + str(name) +  "_" + str(index) +"kekka.jpg",im_con)
	closing = cv2.morphologyEx(im_con, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("work/" + str(name) + "/" + str(name) + "_" + str(index) +"closing10_" + ".jpg", closing)
	print(str(index) + "回目クロージング処理終了")

	return closing

#指先を抽出する処理
def yubisaki(im,name,index):

	height,width = im.shape[:2]
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	#gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)[1]

	#cv2.rectangle(gray, (height-200, height-1), (width-1, height-1), (0, 0, 0), -1)

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
	list_hen = []
	
	a = np.array([y_center,x_center])
	print(str(index) + "回目重心と各画素の距離を算出開始")
	#tt1 = time.time()
	
	
	x=np.arange(width).reshape([1,width])
	y=np.arange(height).reshape([height,1])
	euclidean=((x-a[1])**2+(y-a[0])**2)**0.5
	euclidean = euclidean.reshape((-1))
	euclidean = min_max(euclidean)
	euclidean = euclidean.reshape((height, width))
	ave=np.average(euclidean[dst==max_index]) 

	print("ave",ave)
	euclidean[(dst==max_index)*(euclidean<(2 * ave) -0.1)]=0
	#euclidean[(dst==max_index)*(euclidean< 0.8)]=0
	im[euclidean==0] = [0,0,0]

	"""
	for y in range(height):
		for x in range(width):
			if dst[y][x] == max_index:
				b = np.array([y,x])
				list.append(distance.euclidean(a,b))
	
	

	print(str(index) + "回目重心と各画素の距離を算出終了")
	#ave = np.mean(list)
	#距離の正規化
	list = min_max(list)
	#正規化した距離の平均を出す
	ave = np.average(list)
	print("ave",ave)
	print(str(index) + "回目指先の抽出開始")

	i = 0
	for y in range(height):
		for x in range(width):
			if dst[y][x] == max_index:
				deviation = list[i] - ave
				i += 1
				list2.append(deviation)
				if (deviation) < ave - 0.1:
					im[y][x] = (0,0,0)
	"""
	
	

	txt = str(ave) + "\n"
	f = open('log.txt', 'a') # 書き込みモードで開く
	#f.write(str(name))
	#f.write("\n")
	f.write(str(txt)) # 引数の文字列をファイルに書き込む
	f.close()

	#tt2 = time.time()
	#print(str(index) + "回目2重ループ処理の時間",tt2- tt1)
	#print("指先の抽出終了")
	#ave2 = np.average(list2)
	#su = np.sum(list2)
	cv2.imwrite("work/" + str(name) + "/" + str(name) + str(index) + "_saki" +".jpg", im)

	return im

def main():
	f = open('log.txt', 'a') # 書き込みモードで開く
	now = datetime.datetime.now()
	f.write(str(now))
	f.write("\n")
	"""
	name, ext = os.path.splitext(param[1])
	img = cv2.imread("/Users/dennomaaya/Desktop/py/work/" + str(name) + "/a" + str(param[1]))
	mask(img,name,0)
	"""
	t1 = time.time() 

	p_number = 1
	
	for p_number in range(1,len(param)):
		time1 = time.time()
		img_array = []
		img_array_saki = []

		#画像を読み込む
		image = Image.open("%s"%param[p_number])
		name, ext = os.path.splitext(param[p_number])
		print("ファイル名",name)
		f = open('log.txt', 'a') # 書き込みモードで開く
		f.write(str(name))
		f.write("\n")
		f.close()
		
		#作業領域の生成(recog_handで行うためいらない)
		if os.path.exists("/Users/dennomaaya/Desktop/py/work/" + str(name)) == False:
			os.mkdir("/Users/dennomaaya/Desktop/py/work/" + str(name))
		
		image.save("/Users/dennomaaya/Desktop/py/work/" + str(name) + "/a" + str(param[p_number]),quality=100)
		#cv2.imwrite("/Users/dennomaaya/Desktop/py/work/" + str(name) + "/a" + str(param[p_number]),img)
		img = cv2.imread("/Users/dennomaaya/Desktop/py/work/" + str(name) + "/a" + str(param[p_number]))
		#mask(img,name,0)
		#labelling(mask(img,name,0),name,0)
		
		#画像から手領域の座標を検出
		t_reco1 = time.time()
		print("手が存在する領域の抽出開始")
		hand_range = np.array([recog.hand_detection(param[p_number])])
		t_reco2 = time.time()
		print("手が存在する領域の抽出終了")
		print("手領域探索時間",t_reco2 - t_reco1)



		index = 1
		if hand_range != []:
			while index != hand_range.shape[1]+1:
				#手の範囲の画像を取得する
				ymin, xmin, ymax, xmax = map(int,hand_range[0][index-1])
				img_copy = img[xmin:xmax, ymin:ymax]
				cv2.imwrite("work/" + str(name) + "/" + str(name) + "_" + str(index) +"_cut.jpg", img_copy)

				#画像を配列に格納する
				#img_array.append(img_copy)
				#img_copy = lookup(img_copy,name,index)

				#画像から手だけを抽出する
				gazou = labelling(mask(img_copy,name,index),name,index)

				#手から指先のみを抽出する
				gazou2 = yubisaki(gazou,name,index)

				#指先画像を配列に格納する
				img_array_saki.append(gazou2)
				index += 1

			#指先の指紋を消す
			pt.finger_edit(img, img_array_saki,hand_range,name)
			time2 = time.time()
			print(str(param[p_number]),"の加工時間",time2 - time1)
			print("")
			f = open('log.txt', 'a') # 書き込みモードで開く
			f.write(str("加工時間"))
			f.write(str(time2-time1))
			f.write("\n")
			f.close()
				
		else:
			time2 = time.time()
			no_image = cv2.imread("/Users/dennomaaya/Desktop/py/create/miserarenaiyo.jpg")
			cv2.imwrite("work/" + str(name) + "/a" + str(name) + "print" + ".jpg", no_image)
			print(str(param[p_number]),"は加工しない",time2 - time1)
			print("")

			f = open('log.txt', 'a') # 書き込みモードで開く
			f.write(str("加工時間"))
			f.write(str(time2-time1))
			f.write("\n")
			f.close()
		
		
	t2 = time.time()
	elapsed_time = t2-t1
	print(f"経過時間：{elapsed_time}")
	f = open('log.txt', 'a') # 書き込みモードで開く
	f.write("\n")
	f.close() # ファイルを閉じる
	
if __name__ == '__main__':
	param = sys.argv
	main()