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
    --data_path /cluster/home/akmarala/data/TEXMET \
    --output_dir ./checkpoints_texmet

python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224 \
    --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 64 \
    --data_path /cluster/home/akmarala/data/TEXMET \
    --resume /cluster/home/akmarala/fcvit-mt-ntnu/output_fcvit_simple/FCViT_base_3x3_ep100_lr3e-05_b256.pt

# Fine-tune ImageNet checkpoint on TEXMET
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes 4 \
  main_train.py \
    --backbone vit_base_patch16_224 \
    --size_puzzle 225 \
    --size_fragment 75 \
    --num_fragment 9 \
    --lr 3e-05 \
    --epochs 200 \
    --weight_decay 0.05 \
    --batch_size 64 \
    --num_workers 12 \
    --data_path /cluster/home/akmarala/data/TEXMET \
    --output_dir ./checkpoints_texmet_finetune \
    --resume /cluster/home/akmarala/fcvit-mt-ntnu2/checkpoint-fcvit-imagenet-accelareta/FCViT_base_3x3_ep100_lr3e-05_b64.pt

python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224 \
    --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 64 \
    --data_path /cluster/home/akmarala/data/TEXMET \
    --resume ./checkpoints_texmet_finetune/FCViT_base_3x3_ep200_lr3e-05_b64.pt

python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224     --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 64     --data_path /cluster/home/akmarala/data/TEXMET     --resume ./checkpoints_texmet_finetune2/FCViT_base_3x3_ep600_lr1e-05_b64.pt

python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224 \
    --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 64 \
    --data_path /cluster/home/akmarala/data/inpainting/exp2_regular_masking \
    --resume /cluster/home/akmarala/fcvit-mt-ntnu/output_fcvit_simple/FCViT_base_3x3_ep100_lr3e-05_b256.pt
