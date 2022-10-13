import glob
from operator import ipow
from PIL import Image
import numpy as np
import tqdm
import mmcv
import glob
img_paths = glob.glob('/data/fusai_release/test/images/*.tif')
thre = int(512 * 512 * 0.02)

#测试图片去除蓝色蒙版
def process(img_pth):
    img = Image.open(img_pth)
    ori = np.array(img)
    r, g, b = img.convert("RGB").split()
    r, g, b = r.histogram()[1:], g.histogram()[1:], b.histogram()[1:]
    r_, g_, b_ = np.argmax(r), np.argmax(g), np.argmax(b)
    meanB = np.mean(ori[ori[:, :, 2] != 0, 2])
    stdB = np.std(ori[ori[:, :, 2] != 0, 2])
    delta = b_ - (r_ + g_) // 2
    if delta > 30 and stdB <= 30:
        maxV = 0
        presum = 0
        for i in range(0, 255):
            presum += b[i]
            if presum > thre:
                maxV = i
                break
        # if not (delta < maxV and b_ - maxV > 40):
        ori[:, :, 2] = np.maximum(ori[:, :, 2].astype(np.int32) - min(maxV, delta), 0).astype(np.uint8)
        ori[:, :, 1] = np.maximum(ori[:, :, 1].astype(np.int32) - (g_ - r_) / 4 * 3, 0).astype(np.uint8)
    ori = Image.fromarray(ori)
    ori.save(img_pth.replace('images', 'images2'))
_ = mmcv.track_parallel_progress(process, img_paths, 8)
# process(r"F:\Dataset\InternationalRaceTrackDataset\chusai_release\train\images\1666.tif")
# for i in tqdm.tqdm(img_paths):
#     process(i)
