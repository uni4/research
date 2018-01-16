#https://note.nkmk.me/python-opencv-face-detection-haar-cascade/
import cv2
import numpy as np
import sys
import os.path

def main():
	#images = os.listdir("/Users/dennomaaya/Desktop/py/sei/") #ディレクトリのパスをここに書く
	data_dir_path = u"/Users/dennomaaya/Desktop/py/cascade_directory/sei/"
	file_list = os.listdir(r'/Users/dennomaaya/Desktop/py/cascade_directory/sei/')
	print(file_list)

	for file_name in file_list:
		#root, ext = os.path.splitext(file_name)
		name, ext = os.path.splitext(file_name)
		if ext == u'.png' or u'.jpeg' or u'.jpg' or u'.JPG':
			abs_name = data_dir_path + '/' + file_name
			img = cv2.imread(abs_name)
			height, width = img.shape[:2]
			#name, ext = os.path.splitext(abs_name)
			# 画像の中心座標
			oy, ox = int(height/2), int(width/2)
			i2 = 1

			for rn in range(-30, 30):

				theta = rn # 回転角
				scale = 1.0    # 回転角度・拡大率

				

				#画像の回転
				R = cv2.getRotationMatrix2D((oy, ox), theta, scale)    # 回転変換行列の算出
				dst = cv2.warpAffine(img, R, (width,height), flags=cv2.INTER_CUBIC)
				#dst = cv2.warpAffine(img, R, (height,width), flags=cv2.INTER_CUBIC)
				#dst[dst==[0,0,0]] = [255,255,255]
				dst = np.where(dst==[0,0,0], [255,255,255],dst)
				dst = np.where(dst==[0,0,0], [255,255,255],dst)

				# 結果を出力
				cv2.imwrite("cascade_directory/pos/" + name + "_" + str(i2) + ".jpg", dst)
				print(name + "_" + str(i2) + ".jpg" + "作成完了")
				i2 += 1
			

if __name__ == '__main__':
	param = sys.argv
	main()


