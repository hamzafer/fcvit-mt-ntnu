# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae

# python main_eval.py \
#   --eval \
#   --device cuda:1 \
#   --backbone vit_base_patch16_224 \
#   --size_puzzle 225 \
#   --size_fragment 75 \
#   --num_fragment 9 \
#   --batch_size 256 \
#   --data_path /cluster/home/muhamhz/data/imagenet \
#   --resume /cluster/home/muhamhz/fcvit-mt-ntnu/checkpoint/FCViT_base_3x3_ep100_lr3e-05_b64.pt

# --------------------------------------------------------


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from util.datasets import build_dataset
from puzzle_fcvit import FCViT

from engine_eval import evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('FCViT evaluation for jigsaw puzzle task', add_help=False)
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--device', default='cuda',
                        help='device to use for evaluation')

    # Model parameters
    parser.add_argument('--backbone', default='vit_base_patch16_224', type=str, metavar='MODEL')
    parser.add_argument('--size_puzzle', default=225, type=int,
                        help='puzzle image size')
    parser.add_argument('--size_fragment', default=75, type=int,
                        help='fragment of puzzle image size')
    parser.add_argument('--num_fragment', default=9, type=int,
                        help='fragment of puzzle image size')

    # Dataset parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--data_path', default='./data/ImageNet/', type=str,
                        help='dataset path')
    return parser


def main(args):
    device = torch.device(args.device)
    # device = 'cpu'

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_val = build_dataset(is_train=False, args=args)

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = FCViT(
        backbone=args.backbone,
        num_fragment=args.num_fragment,
        size_fragment=args.size_fragment
    )
    model.to(device)
    model.augment_fragment = transforms.Compose([
        transforms.RandomCrop(round(args.size_fragment * 0.85)),
        transforms.Resize((args.size_fragment, args.size_fragment), antialias=True),
    ])
    epochs = [0]
    if len(args.resume) == 0:
        print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Epoch: {epochs[-1]}')
        print(f'WARNING: The model is untrained.')
        print(f'WARNING: You need to check resume of args.')
    else:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        epochs = checkpoint['epochs']
        clean_state = {k.replace('module.', '', 1): v
                    for k, v in checkpoint['model'].items()}
        model.load_state_dict(clean_state, strict=True)
        print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Epoch: {epochs[-1]}')

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy (Fragment-level) of the network on the {len(dataset_val)} test images: {test_stats['acc_fragment']:.2f}%")
        print(f"Accuracy (Puzzle-level) of the network on the {len(dataset_val)} test images: {test_stats['acc_puzzle']:.2f}%")
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
