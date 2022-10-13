source activate mmseg
cd /data/mmseg/C_run
nohup python mmseg/tools/test.py convnext_val_vis.py \
 work_dirs/convnext_l_lova_aug/latest.pth \
 --show-dir conv_val_aug > visual.log &