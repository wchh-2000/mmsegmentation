source activate mmlab
cd /data/mmseg/C_run
PORT=${PORT:-29501}
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ../mmseg/tools/test.py ../C_cfg/convnext_l_60e_lova_aug.py \
    ../C_work_dirs/convnext_l_lova_aug/latest.pth \
     --format-only --eval-options "imgfile_prefix=./results" \
     --launcher pytorch ${@:4} > test.log &
     #--eval "mIoU"