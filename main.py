# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from util.datasets import build_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from puzzle_fcvit import FCViT


def get_args_parser():
    parser = argparse.ArgumentParser('FCViT training for 3x3 jigsaw puzzle task', add_help=False)
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--device', default='cuda',
                        help='device to use for evaluation')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL')
    parser.add_argument('--puzzle_size', default=225, type=int,
                        help='puzzle image size')
    parser.add_argument('--fragment_size', default=75, type=int,
                        help='fragment of puzzle image size')

    # Dataset parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--data_path', default='../datasets/ImageNet/', type=str,
                        help='dataset path')
    return parser


def main(args):
    device = torch.device(args.device)

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

    model = FCViT().to(device)
    model.augment_fragment = transforms.Compose([
        transforms.RandomCrop(round(args.fragment_size * 0.85)),
        transforms.Resize((args.fragment_size, args.fragment_size)),
    ])
    epochs = [0]
    losses = [0]
    accuracies = [0]
    if len(args.resume) == 0:
        print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Epoch: {epochs[-1]}')
        print(f'WARNING: The model is untrained.')
        print(f'WARNING: You need to check resume of args.')
    else:
        checkpoint = torch.load(args.resume)
        epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['model'])
        losses = checkpoint['losses']
        print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Epoch: {epochs[-1]}')

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    ''' ############################ 여기까지 수정 완 ############################'''


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
