from PIL import Image
import numpy as np
import mmcv
import os
import os.path as osp
import warnings
from tqdm import tqdm
import random
#初赛：
# CLASSES = ('background', 'water','transport', 'building','agricultural','grass', 'forest','barren','others'
#             )

# PALETTE = [[0, 60, 100], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
#             [159, 129, 183], [0, 255, 0], [255, 195, 128], [0, 0, 0]]
CLASSES = ['Background','Waters', 'Road', 'Construction', 'Airport', 'Railway Station',
    'Photovoltaic panels', 'Parking Lot', 'Playground','Farmland', 'Greenhouse', 'Grass',
    'Artificial grass', 'Forest', 'Artificial forest', 'Bare soil', 'Artificial bare soil', 'Other']
PALETTE = [[0,0,0], [0,0,255], [128,128,128], [255,127,131],[192,192,192], [255,255,255],
    [79,100,118], [70,70,70],[255,73,73], [255,255,0], [60,177,246], [137,35,245],
        [205,133,250], [0,114,8], [102,216,103], [128,64,3], [255,128,0],[255,51,205]]
def show_result(data):
    """Draw `result` over `img`.
    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img,result,out_file=data
    palette = PALETTE
    opacity=0.5
    img = mmcv.imread(img)
    img = img.copy()
    seg = np.array(Image.open(result))
    palette = np.array(palette)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    mmcv.imwrite(img, out_file)

def myImread(path):
    file_client = mmcv.FileClient(backend='disk')
    img_bytes = file_client.get(path)
    img = mmcv.imfrombytes(
        img_bytes, flag='unchanged',
        backend='pillow').squeeze().astype(np.int32)
    # rough classes
    for i in range(9):
        img[img // 100 == i] = i
    # fine classes
    # for i in range(1, 17):
    #     gt_semantic_seg[gt_semantic_seg % 100 == i] = i
    # gt_semantic_seg[gt_semantic_seg % 100 == 17] = 0
    return img.astype(np.uint8)

if __name__ == '__main__':
    labelPth = "/data/fusai_release/train/labels_18/"
    imgPth = "/data/fusai_release/train/images/"
    savePth = "/data/fusai_release/visual_labels/"
    data=[(imgPth+n.split('.')[0]+'.tif',labelPth+n,savePth+n) for n in os.listdir(labelPth)]
    mmcv.track_parallel_progress(show_result,data,12)
