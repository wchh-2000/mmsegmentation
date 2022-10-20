from PIL import Image
import numpy as np
import mmcv
import os
import os.path as osp
from tqdm import tqdm
import random
def get_std_mean(data: np.ndarray, ratio: int = 1):
    '''
    function: get the mean and standard deviation using different proporations of data
    params:
        data(numpy.ndarray, N*H*W*C): image data
        ratio(float): the proportion of data used to calculate
    '''
    data_num = len(data)
    idx = list(range(data_num))
    random.shuffle(idx)
    end = int(ratio * data_num)
    if end <= 0:
        end = 1
    data_selected = data[idx[0:end]]
    print(data_selected.dtype)
    # data_selected = data_selected.astype(np.float32)
    mean = np.mean(data_selected, axis=(0, 1, 2)) #/ 255
    std = np.std(data_selected, axis=(0, 1, 2)) #/ 255
    return mean.tolist(), std.tolist()
if __name__ == '__main__':
    imgPath = '/data/fusai_release/train/images'
    data = []
    for file in tqdm(os.listdir(imgPath)):
        img = mmcv.imread(osp.join(imgPath, file))
        # if data is None:
        #     data = img
        # else:
        #     data = np.stack(data, img, axis=0)
        data.append(img)
        break
    data = np.array(data)
    print(get_std_mean(data, 1))