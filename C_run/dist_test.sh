source activate mmseg
cd /data/mmseg/C_run
PORT=${PORT:-29501}
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ../tools/test.py ../C_cfg/convnext_s_60e.py \
    ../C_work_dirs/conv_s_lova/epoch_40.pth \
     --format-only --eval-options "imgfile_prefix=./results" \
     --launcher pytorch ${@:4} > test.log &
     #--eval "mIoU"