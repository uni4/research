#python3 print.py piece1.JPG test.jpg 
#作成した指先画像から指先を編集する
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image

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

#重心を計算するモジュール	
def gravity(image,p_number):
	image = cv2.imread("closing10_" + str(p_number) + ".jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mu = cv2.moments(gray, False)
	x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
	return x,y

def finger_edit(im,im_fil,hand_range,name):
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
		#cv2.circle(image, (int(center[max_index][0]),int(center[max_index][1])),10,(255,0,0),-1)
		
		#指先を染めるための色情報を作る
		#flesh = image[int(center[max_index][0]),int(center[max_index][1])]
		
		print("指紋部分の処理開始")
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
			#img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, c_kernel)
			img_cc = addweight(img_copy,1,0.5,0)
			img_cc = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, c_kernel)
			#gray2 = cv2.threshold(filter_copy, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			#ist_mask = cv2.calcHist([img_copy],[0],filter_copy,[256],[0,256])
			#plt.subplot(index + 221), plt.plot(hist_mask)

			#肌色の部分を変換する
			for y_index in range(height):
				for x_index in range(width):
					blue = img_mask[y_index, x_index, 0]
					green = img_mask[y_index, x_index, 1]
					red = img_mask[y_index, x_index, 2]
					b1 = img_cc[y_index, x_index,0] 
					g1 = img_cc[y_index, x_index,1]
					r1 = img_cc[y_index, x_index,2]  
					if blue != 0 and green != 0 and red != 0 and b1 != 0 and g1 != 0 and r1 != 0:
						#img_copy[y_index, x_index] = img_cc[y_index, x_index]
						img_copy[y_index, x_index] = [0,0,255]
			cv2.imwrite("work/" + str(name) + "/" + str(name) + "range_" + str(index+1) +".jpg", img_copy)
			cv2.imwrite("work/" + str(name) + "/" + str(name) + "mask_" + str(index+1) +".jpg", filter_copy)

	print("指紋部分の処理終了")
	cv2.imwrite("work/" + str(name) + "/a" + str(name) + "print" + ".jpg", im,[cv2.IMWRITE_JPEG_QUALITY,100])
	#plt.xlim([0,256])
	#plt.show()

def main():
	image = cv2.imread("%s"%param[1])
	im_filter = cv2.imread("%s"%param[2])
	finger_edit(param,im_filter,1)

if __name__ == '__main__':
	param = sys.argv
	main()