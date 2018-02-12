#python3 print.py piece1.JPG test.jpg 
#作成した指先画像から指先を編集する
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import time

#画像を合成するモジュール
def addweight(img,alpha,beta,ganma):	
	gauss = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3)
	#おそらくラプラシアンフィルタ
	kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
      
	himg2 = cv2.filter2D(img, -1, kernel)
	im_add = cv2.addWeighted(gauss, alpha, himg2, beta, ganma)
	return im_add

#画像を合成するモジュール
def addweight2(img,alpha,beta,ganma):
	height, width = img.shape[:2]
	shimon = cv2.imread("s01.jpg")
	shimon = cv2.resize(shimon,(width,height))
	im_add = cv2.addWeighted(img, alpha, shimon, beta, ganma)
	cv2.imwrite("s.jpg",im_add)
	return im_add

#重心を計算するモジュール	
def gravity(image,p_number):
	image = cv2.imread("closing10_" + str(p_number) + ".jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mu = cv2.moments(gray, False)
	x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
	return x,y

def finger_edit(im,im_fil,hand_range,name):
	print("指紋部分の処理開始")
	tt1 = time.time()
	for i in range(hand_range.shape[1]):
		ymin, xmin, ymax, xmax = map(int,hand_range[0][i])
		image = im[xmin:xmax, ymin:ymax]
		im_filter = im_fil[i]
		c_kernel = np.ones((7,7),np.uint8)

		#フィルターを2値化してラベリングする
		gray = cv2.cvtColor(im_filter, cv2.COLOR_BGR2GRAY)
		gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		label = cv2.connectedComponentsWithStats(gray)

		
		# ブロブ情報を項目別に抽出
		n = label[0] - 1
		data = np.delete(label[2], 0, 0)
		center = np.delete(label[3], 0, 0)
		max_index = np.argsort(data[:,4])[::-1][0] + 1
		
		for index in range(n):
			x = data[index,0]
			y = data[index,1]
			w = data[index,2]
			h = data[index,3]
			img_copy = image[y:y+h, x:x+w]
			filter_copy = im_filter[y:y+h, x:x+w]
			filter_copy = cv2.cvtColor(filter_copy, cv2.COLOR_BGR2GRAY)
			filter_copy = cv2.threshold(filter_copy, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			height, width = img_copy.shape[:2]

			#ヒストグラムを作成する
			#hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
			#lower = np.array([0, 30, 60])
			#upper = np.array([30, 150, 255])
			#mask = cv2.inRange(hsv, lower, upper)
			img_mask = cv2.bitwise_and(img_copy, img_copy, mask=filter_copy)
			#img_cc = addweight(img_copy,1,0.5,0)
			#img_cc = addweight2(img_copy,0.5,1,0)
			img_cc = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, c_kernel)

			#肌色の部分を変換する
			img_mask_bgr = cv2.split(img_mask)
			img_cc_bgr = cv2.split(img_cc)
			blue = img_mask_bgr[0]
			green = img_mask_bgr[1]
			red = img_mask_bgr[2]
			#img_copy[blue != 0 and green != 0 and red!= 0\
			#		 and b1 != 0 and g1 != 0 and r1 != 0] = img_cc
			
			for y_index in range(height):
				for x_index in range(width):					
					if blue[y_index, x_index] != 0 and green[y_index, x_index] != 0 and red[y_index, x_index] != 0:
						#img_copy[y_index, x_index] = img_cc[y_index, x_index]
						img_copy[y_index, x_index] = [0,0,255]
			
			cv2.imwrite("work/" + str(name) + "/" + str(name) + "range_" + str(index+1) +".jpg", img_copy)
			cv2.imwrite("work/" + str(name) + "/" + str(name) + "mask_" + str(index+1) +".jpg", filter_copy)

	print("指紋部分の処理終了")
	tt2 = time.time()
	print("指紋部分の処理時間",tt2- tt1)
	cv2.imwrite("work/" + str(name) + "/a" + str(name) + "print" + ".jpg", im,[cv2.IMWRITE_JPEG_QUALITY,100])


def main():
	image = cv2.imread("%s"%param[1])
	im_filter = cv2.imread("%s"%param[2])
	finger_edit(param,im_filter,1)

if __name__ == '__main__':
	param = sys.argv
	main()