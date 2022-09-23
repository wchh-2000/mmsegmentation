source activate mmLab
cd /data/mmseg/C_run
# nohup python ../tools/test.py ../C_cfg/convnext_b_60e_lova_aug.py \
#  ../C_work_dirs/convnext_b_lova_aug/latest.pth --gpu-id 0 \
#  --format-only --eval-options "imgfile_prefix=./results" > test.log &
# --show-dir test_visualization --eval "mIoU" 

# nohup python ../tools/test.py ../C_cfg/convnext_l_60e_lova_aug.py \
#  ../C_work_dirs/convnext_l_lova_aug/latest.pth --gpu-id 1 \
#  --format-only --eval-options "imgfile_prefix=./results" > test.log &


 nohup python ../tools/test_inte.py /data/mmseg/C_cfg/integrate_test_cfg.py \
/data/mmseg/C_work_dirs/convnext_l_lova_aug/epoch_60_best.pth \
/data/mmseg/C_work_dirs/convNeXt/iter_40000.pth --gpu-id 0 \
 --format-only --eval-options "imgfile_prefix=/data/integrate_results" >> integrate_results.log &