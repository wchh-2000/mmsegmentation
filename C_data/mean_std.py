import time
import mmcv
import numpy as np
img = mmcv.imread('/data/fusai_release/train/images/10.tif')
# print(img.shape)
# s=time.time()
mean = np.mean(img, axis=(0, 1))
std = np.std(img, axis=(0, 1))
# print(time.time()-s)
print(mean,std)