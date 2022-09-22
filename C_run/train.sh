source activate mmLab
cd /data/mmseg/C_run
# nohup python ../tools/train.py  ../C_cfg/convnext_b_60e_lova_aug.py \
#  --work-dir ../C_work_dirs/convnext_b_lova_aug --gpu-id 0  \
#  --seed 779621588 > base.log &
nohup python ../tools/train.py  ../C_cfg/convnext_l_60e_lova_aug.py \
 --work-dir ../C_work_dirs/convnext_l_lova_aug --gpu-id 1 \
 --seed 119032369 > large.log &
#--no-validate 
# --load-from work_dirs/convNeXt/iter_40000.pth 
# --resume-from /data/work_dir_9classes/iter_65600.pth
# pid 23209