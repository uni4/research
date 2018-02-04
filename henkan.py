#https://note.nkmk.me/python-opencv-face-detection-haar-cascade/
import cv2
import numpy as np
import sys
import os.path

def main():
	#images = os.listdir("/Users/dennomaaya/Desktop/py/sei/") #ディレクトリのパスをここに書く
	data_dir_path = u"/Users/dennomaaya/Desktop/py/cascade_directory/seikai"
	file_list = os.listdir(r'/Users/dennomaaya/Desktop/py/cascade_directory/seikai')
	i2 = 311

	for file_name in file_list:
		name, ext = os.path.splitext(file_name)
		if ext == u'.png' or u'.jpeg' or u'.jpg' or u'.JPG' and name == ".DS_Store":
			abs_name = data_dir_path + '/' + file_name
			img = cv2.imread(abs_name)
			#dst2 = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR)
			# 結果を出力
			cv2.imwrite("create/photo/" + str(i2) + ".jpg", img,[cv2.IMWRITE_JPEG_QUALITY,100])
			print(file_name + ".jpg" + "変換完了")
			i2 += 1


if __name__ == "__main__":
	main()