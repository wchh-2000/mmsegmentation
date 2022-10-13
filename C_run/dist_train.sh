source activate mmseg
cd /data/mmseg/C_run
PORT=${PORT:-29500}
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
  ../tools/train.py ../C_cfg/convnext_s_60e.py --work-dir ../C_work_dirs/convnext_s_dist \
  --seed 806364387 --launcher pytorch ${@:3} > train.log &
  #--no-validate 