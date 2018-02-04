from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('lena_square.png')

plt.figure()
r = np.array(im)[:, :, 0].flatten()
g = np.array(im)[:, :, 1].flatten()
b = np.array(im)[:, :, 2].flatten()

bins_range = range(0, 257, 8)
xtics_range = range(0, 257, 32)

plt.hist((r, g, b), bins=bins_range,
        color=['r', 'g', 'b'], label=['Red', 'Green', 'Blue'])
ret = np.histogram((r, g, b), bins=bins_range)
plt.legend(loc=2)

plt.grid(True)

[xmin, xmax, ymin, ymax] = plt.axis()
plt.axis([0, 256, 0, ymax])
plt.xticks(xtics_range)
plt.savefig("matplotlib_histogram_single.png")
print("度数",ret[0])
print("階級値",ret[1])