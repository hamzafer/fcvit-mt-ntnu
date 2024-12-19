# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import argparse
import datetime
import time
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from util.datasets import build_dataset
from puzzle_fcvit import FCViT

from engine_train import training


def get_args_parser():
    parser = argparse.ArgumentParser('FCViT training for jigsaw puzzle task', add_help=False)
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

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=3e-05, metavar='LR',
                        help='learning rate (default: 3e-05)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Dataset parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--data_path', default='./data/ImageNet/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./save',
                        help='path where to save, empty for no saving')
    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train = build_dataset(is_train=True, args=args)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    dataset_val = Subset(dataset_train, list(range(int(0.1 * len(dataset_train)))))  # 0.01
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = FCViT(
        backbone=args.backbone,
        num_fragment=args.num_fragment,
        size_fragment=args.size_fragment
    )
    model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    epochs = [0]
    losses = [0]
    accuracies = [0]
    if len(args.resume) == 0:
        print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Epoch: {epochs[-1]}')
    else:
        checkpoint = torch.load(args.resume)
        epochs = checkpoint['epochs']
        losses = checkpoint['losses']
        accuracies = checkpoint['accuracies']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            temp_optim = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            temp_scheduler = CosineAnnealingLR(temp_optim, T_max=args.epochs)
            [temp_scheduler.step() for _ in range(checkpoint['epochs'][-1])]
            scheduler.load_state_dict(temp_scheduler.state_dict())
        args.start_epoch = epochs[-1]
        print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Epoch: {epochs[-1]}')
    print(optimizer)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_stats = training(
        model, data_loader_train, data_loader_val, device,
        criterion, optimizer, scheduler,
        epochs, losses, accuracies,
        args=args
    )
    print(f"Training on the {len(dataset_train)} train images, {train_stats['epoch']} epochs,")
    print(f"Loss: {train_stats['loss']:.4f}")
    print(f"Accuracy (fragment-level): {train_stats['accuracy']:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

    '''########################### 여기까지 완료 ###########################'''
    '''
    세이브 패스 잘못 됐음 수정 필요
    '''


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
