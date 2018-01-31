import numpy as np
from scipy.spatial import distance

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def main():
	a = np.array([1,2,3,2,1])

	z_score = zscore(a)
	z_sum1 = z_score[0] + z_score[1] + z_score[2] + z_score[3] + z_score[4]
	z_sum2 = np.sum(z_score)
	z_ave = np.average(z_score)

	print("zscore",z_score)
	print("zスコアの合計",z_sum1)
	print("zスコアの合計",z_sum2)
	print("zスコアの平均",z_ave)

if __name__ == '__main__':
	main()