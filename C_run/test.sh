source activate mmLab
cd /data/mmseg/C_run
nohup python mmseg/tools/test.py ../C_cfg/convnext_l_60e_lova_aug.py \
 ../C_work_dirs/convnext_l_lova_aug/latest.pth --gpu-id 0 \
 --format-only --eval-options "imgfile_prefix=./results" > test.log &
# --show-dir test_visualization --eval "mIoU" 

# nohup python mmseg/tools/test.py convnext_l_60e_lova_aug.py \
#  work_dirs/convnext_l_lova_aug/latest.pth --gpu-id 1 \
#  --format-only --eval-options "imgfile_prefix=./results" > test.log &