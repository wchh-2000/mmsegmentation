import glob
from PIL import Image
import numpy as np
import mmcv

import glob
label_paths = glob.glob('/data/chusai_release/train/labels_9/*.png')
def process(label_path):
    label = np.array(Image.open(label_path))
    label = np.where(label==2,255,0)
    label = label.astype(np.uint8)
    label = Image.fromarray(label)
    label.save(label_path.replace('train/labels_9', 'road/gt'))
_ = mmcv.track_parallel_progress(process, label_paths, 12)
# process('/data/chusai_release/train/labels_9/6.png')