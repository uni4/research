#python3 print.py piece1.JPG test.jpg 
#作成した指先画像から指先を編集する
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
	image = cv2.imread("%s"%param[1])
	im_filter = cv2.imread("%s"%param[2])

	#フィルターを2値化してラベリングする
	gray = cv2.cvtColor(im_filter, cv2.COLOR_BGR2GRAY)
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	label = cv2.connectedComponentsWithStats(gray)

	# ブロブ情報を項目別に抽出
	n = label[0] - 1
	data = np.delete(label[2], 0, 0)
	center = np.delete(label[3], 0, 0)
	max_index = np.argsort(data[:,4])[::-1][0] + 1

	#指先を染めるための色情報を作る
	flesh = image[int(center[max_index][0]),int(center[max_index][1])]

	print("最大面積のラベル番号", max_index)
	print("ブロブの個数:", n)
	print("各ブロブの外接矩形の左上x座標", data[:,0])
	print("各ブロブの外接矩形の左上y座標", data[:,1])
	print("各ブロブの外接矩形の幅", data[:,2])
	print("各ブロブの外接矩形の高さ", data[:,3])
	print("各ブロブの面積", data[:,4])
	print("中心座標:\n",center[max_index])
	

	for index in range(n):
		x = data[index,0]
		y = data[index,1]
		w = data[index,2]
		h = data[index,3]
		img_copy = image[y:y+h, x:x+w]
		height, width = img_copy.shape[:2]

		#ヒストグラムを作成する
		hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
		lower = np.array([0, 30, 60])
		upper = np.array([20, 150, 255])
		mask = cv2.inRange(hsv, lower, upper)
		img_mask = cv2.bitwise_and(img_copy, img_copy, mask=mask)
		gray2 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		hist_mask = cv2.calcHist([img_copy],[0],mask,[256],[0,256])
		plt.subplot(index + 221), plt.plot(hist_mask)

		#肌色の部分を変換する
		for y_index in range(height):
			for x_index in range(width):
				blue = img_mask[y_index, x_index, 0]
				green = img_mask[y_index, x_index, 1]
				red = img_mask[y_index, x_index, 2]
				if blue != 0 and green != 0 and red != 0:
					img_copy[y_index, x_index] = img_copy[y_index, x_index]

		cv2.imwrite("range_" + str(index+1) +".jpg", img_copy)

	cv2.imwrite("print.jpg", image)
	plt.xlim([0,256])
	#plt.show()

if __name__ == '__main__':
	param = sys.argv
	main()