python main_train.py --backbone vit_base_patch16_224 --epochs 100
python main_train.py --backbone vit_base_patch16_224 --epochs 100 --wandb_offline
python main_train.py --backbone vit_base_patch16_224 --epochs 100 --disable_wandb
python main_train.py --backbone vit_base_patch16_224 --epochs 100 --wandb_run_name "my_experiment_v1"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes 4 \
  main_train.py \
    --backbone vit_base_patch16_224 \
    --size_puzzle 225 \
    --size_fragment 75 \
    --num_fragment 9 \
    --lr 3e-05 \
    --epochs 500 \
    --weight_decay 0.05 \
    --batch_size 64 \
    --num_workers 12 \
    --data_path /cluster/home/akmarala/data/imagenet \
    --output_dir /cluster/home/akmarala/fcvit-mt-ntnu/output_fcvit_newtestt