# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Seg18Dataset(CustomDataset):
    """
    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ['Background','Waters', 'Road', 'Construction', 'Airport', 'Railway Station',
    'Photovoltaic panels', 'Parking Lot', 'Playground','Farmland', 'Greenhouse', 'Grass',
    'Artificial grass', 'Forest', 'Artificial forest', 'Bare soil', 'Artificial bare soil', 'Other']
    PALETTE = [[0,0,0], [0,0,255], [128,128,128], [255,127,131],[192,192,192], [255,255,255],
     [79,100,118], [70,70,70],[255,73,73], [255,255,0], [60,177,246], [137,35,245],
            [205,133,250], [0,114,8], [102,216,103], [128,64,3], [255,128,0],[255,51,205]]

    def __init__(self,cal_mean_std=False,load_mean_std=False, **kwargs):
        super(Seg18Dataset, self).__init__(
            **kwargs)
        self.cal_mean_std=cal_mean_std
        self.load_mean_std=load_mean_std
        if cal_mean_std:
            self.normalize_para = {}
            fileClient = mmcv.FileClient(backend='disk')
            Mean=np.zeros((len(self.img_infos),3))
            Std=np.zeros((len(self.img_infos),3))
            for i,dic in enumerate(self.img_infos):
                filepth = osp.join(self.img_dir, dic['filename'])
                img_bytes = fileClient.get(filepth)
                img = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
                mean,std=np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
                mean[0],mean[2]=mean[2],mean[0]#bgr转rgb
                std[0],std[2]=std[2],std[0]
                Mean[i,:]=mean
                Std[i,:]=std
                self.normalize_para[dic['filename']] = [mean,std]
            if self.test_mode:
                w='test'
            else:
                w='train'
            np.savez(f'/data/mmseg/C_data/mean_std_{w}.npz',Mean,Std)
            print("="*5+"calculate mean std done"+"="*5)
        if load_mean_std:
            self.normalize_para = {}
            if self.test_mode:
                w='test'
            else:
                w='train'
            t=np.load(f'/data/mmseg/C_data/mean_std_{w}.npz')
            Mean,Std=t['arr_0'],t['arr_1']
            for i,dic in enumerate(self.img_infos):
                self.normalize_para[dic['filename']] = [Mean[i],Std[i]]
            print("="*5+"load mean std done"+"="*5)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.cal_mean_std or self.load_mean_std:
            results['normalize_para'] = self.normalize_para[results['img_info']['filename']]#优化
        if self.custom_classes:
            results['label_map'] = self.label_map
    
    def results2img(self, results, imgfile_prefix, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            lut=np.array([0,1,2,3,2,2,8,8,8,4,4,5,5,6,6,7,7,8])
            result=result+100*lut[result]#百位加上一级类别
            output = Image.fromarray(result.astype(np.int32))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self, results, imgfile_prefix, indices=None):
        """Format the results into dir (standard format for LoveDA evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, indices)

        return result_files
