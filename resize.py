#https://note.nkmk.me/python-opencv-face-detection-haar-cascade/
import cv2
import numpy as np
import sys
import os.path

def main():
	#images = os.listdir("/Users/dennomaaya/Desktop/py/sei/") #ディレクトリのパスをここに書く
	data_dir_path = u"/Users/dennomaaya/Desktop/py/cascade_directory/batu_moto"
	file_list = os.listdir(r'/Users/dennomaaya/Desktop/py/cascade_directory/batu_moto')
	i2 = 1

	for file_name in file_list:
		name, ext = os.path.splitext(file_name)
		if ext == u'.png' or u'.jpeg' or u'.jpg' or u'.JPG' and name != ".DS_Store":
			abs_name = data_dir_path + '/' + file_name
			img = cv2.imread(abs_name)
			height, width = img.shape[:2]
			# 方法2(OpenCV)
			dst2 = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
			#dst2 = cv2.resize(img, (300, 300))
			# 結果を出力
			cv2.imwrite("cascade_directory/neg/" + str(i2) + ".jpg", dst2)
			print(file_name + ".jpg" + "リサイズ完了")
			i2 += 1


if __name__ == "__main__":
	main()