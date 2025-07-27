# References:
# DeiT: https://github.com/facebookresearch/deit
# # MAE: https://github.com/facebookresearch/mae
# first 10 epcoh: 
# Accuracy (Fragment-level) of the network on the 50000 test images: 12.17%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 0.00%
# first 20
# Accuracy (Fragment-level) of the network on the 50000 test images: 14.33%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 0.00%
# first 30
# Accuracy (Fragment-level) of the network on the 50000 test images: 42.03%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 12.09%
# first 40
# Accuracy (Fragment-level) of the network on the 50000 test images: 63.97%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 36.42%
# first 50
# Accuracy (Fragment-level) of the network on the 50000 test images: 73.04%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 49.68%
# first 70
# Accuracy (Fragment-level) of the network on the 50000 test images: 80.50%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 62.07%
# first 90
# Accuracy (Fragment-level) of the network on the 50000 test images: 83.50%
# Accuracy (Puzzle-level) of the network on the 50000 test images: 68.07%
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
        checkpoint = torch.load(args.resume, weights_only=False)
        epochs = checkpoint['epochs']
        
        # Handle module prefix mismatch
        state_dict = checkpoint['model']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                # Remove 'module.' prefix
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
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
