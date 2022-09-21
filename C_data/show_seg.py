from PIL import Image
import numpy as np
label = np.array(Image.open("/data/test_results/0.png"))
print(len(label),len(label[0]))

from struct import pack
import numpy as np
import mmcv
import os
import os.path as osp
import warnings
from tqdm import tqdm
import random

CLASSES = ('background', 'water','transport', 'building','agricultural','grass', 'forest','barren','others'
            )

PALETTE = [[0, 60, 100], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128], [0, 0, 0]]

def show_result(img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result
        if palette is None:
            if PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

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
    mean = np.mean(data_selected, axis=(0, 1, 2)) / 255
    std = np.std(data_selected, axis=(0, 1, 2)) / 255
    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    labelPath = r'F:\Dataset\InternationalRaceTrackDataset\chusai_release\train\labels'
    imgPath = r'F:\Dataset\InternationalRaceTrackDataset\chusai_release\train\images'
    savePath = r'F:\Dataset\InternationalRaceTrackDataset\chusai_release\train\visual_results'
    for file in tqdm(os.listdir(labelPath)):
        img = mmcv.imread(osp.join(imgPath, file.split('.')[0] + '.tif'))
        label = myImread(osp.join(labelPath, file))
        show_result(img=img, result=label, out_file=osp.join(savePath, file.split('.')[0] + '.tif'))

    # imgPath = r'/data/chusai_release/train/images'
    # data = []
    # for file in tqdm(os.listdir(imgPath)):
    #     img = mmcv.imread(osp.join(imgPath, file))
    #     # if data is None:
    #     #     data = img
    #     # else:
    #     #     data = np.stack(data, img, axis=0)
    #     data.append(img)
    #     # break
    # data = np.array(data)
    # print(get_std_mean(data, 0.8))