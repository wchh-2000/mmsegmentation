source activate mmlab
cd /data/mmseg/C_run
PORT=${PORT:-29500}
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
  mmseg/tools/train.py convnext_l_bs32_40k_ms.py --work-dir work_dirs/convNext_dist \
  --seed 806364387 --launcher pytorch ${@:3} > train.log &
  #--no-validate 