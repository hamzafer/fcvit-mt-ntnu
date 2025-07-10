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
from accelerate import Accelerator                     # ★ NEW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import wandb

from util.datasets import build_dataset
from puzzle_fcvit import FCViT

from engine_train import training

from torchvision.datasets import FakeData
from torchvision import transforms

# ----------------------------------------------------------------------
# argument parser
# ----------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser(
        "FCViT training for jigsaw‑puzzle task", add_help=False
    )

    # GPU / device
    parser.add_argument("--device", default="cuda", help="device to use for training")

    # Model
    parser.add_argument("--backbone", default="vit_base_patch16_224", type=str)
    parser.add_argument("--size_puzzle", default=225, type=int)
    parser.add_argument("--size_fragment", default=75, type=int)
    parser.add_argument("--num_fragment", default=9, type=int)

    # Optimiser / schedule
    parser.add_argument("--lr", type=float, default=3e-05, metavar="LR")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # Data
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="", help="path to checkpoint")
    parser.add_argument("--data_path", default="./data/ImageNet/", type=str)
    parser.add_argument("--output_dir", default="./save", type=str)

    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a 1‑epoch, 2‑GPU, synthetic‑data test")

    # WandB parameters
    parser.add_argument("--wandb_offline", action="store_true",
                        help="Run wandb in offline mode")
    parser.add_argument("--wandb_run_name", default=None, type=str,
                        help="wandb run name (auto-generated if not provided)")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable wandb logging entirely")

    return parser


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(args):
    print("job dir:", os.path.dirname(os.path.realpath(__file__)))
    print(vars(args))

    # ── Accelerate initialisation ─────────────────────────────────────
    accelerator = Accelerator()                      # ★ NEW
    device = accelerator.device                      # ★ NEW

    # ── Initialize wandb ──────────────────────────────────────────────
    if not args.disable_wandb and accelerator.is_main_process:
        # Set wandb mode
        wandb_mode = "offline" if args.wandb_offline else "online"
        
        # Create run name if not provided
        if args.wandb_run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract dataset name from data_path
            dataset_name = "unknown"
            if args.data_path:
                if "imagenet" in args.data_path.lower():
                    dataset_name = "imagenet"
                elif "coco" in args.data_path.lower():
                    dataset_name = "coco"
                elif "places" in args.data_path.lower():
                    dataset_name = "places"
                elif "fake" in args.data_path.lower() or args.smoke_test:
                    dataset_name = "fake"
                else:
                    # Extract last folder name as dataset name
                    dataset_name = os.path.basename(args.data_path.rstrip('/'))
            
            if args.smoke_test:
                dataset_name = "smoke_test"
            
            # Create descriptive run name
            backbone_short = args.backbone.replace("vit_", "").replace("_patch16_224", "")
            fragments_str = f"{args.num_fragment}frag"
            puzzle_size = f"{args.size_puzzle}px"
            batch_lr = f"bs{args.batch_size}_lr{args.lr}"
            
            args.wandb_run_name = f"{dataset_name}_{backbone_short}_{fragments_str}_{puzzle_size}_{batch_lr}_{timestamp}"
        
        # Initialize wandb with your project
        wandb_run = wandb.init(
            project="fcvit",
            entity="hamzafer3-ntnu",
            name=args.wandb_run_name,
            config=vars(args),
            mode=wandb_mode,
            tags=[
                args.backbone, 
                f"{args.num_fragment}fragments", 
                "jigsaw-puzzle",
                dataset_name,
                "smoke-test" if args.smoke_test else "full-training"
            ]
        )
        
        print(f"WandB initialized in {wandb_mode} mode")
        print(f"Run name: {args.wandb_run_name}")
        print(f"Project: https://wandb.ai/hamzafer3-ntnu/fcvit")
        print(f"Direct run link: https://wandb.ai/hamzafer3-ntnu/fcvit/runs/{wandb_run.id}")

    # deterministic seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── datasets & dataloaders ────────────────────────────────────────
    if args.smoke_test:
        fake_transform = transforms.Compose([
            transforms.Resize((args.size_puzzle, args.size_puzzle)),
            transforms.ToTensor(),
        ])
        dataset_train = FakeData(
            size=512, image_size=(3, args.size_puzzle, args.size_puzzle),
            num_classes=1000, transform=fake_transform
        )
    else:
        dataset_train = build_dataset(is_train=True, args=args)

    # small validation split (10 %) or reuse train set in smoke‑test
    if args.smoke_test:                               # ★ NEW
        dataset_val = dataset_train                   # ★ NEW
    else:                                             # ★ NEW
        dataset_val = Subset(dataset_train, list(range(int(0.1 * len(dataset_train)))))  # ★ MOVED

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # ── model & optimisation objects ─────────────────────────────────-
    model = FCViT(
        backbone=args.backbone,
        num_fragment=args.num_fragment,
        size_fragment=args.size_fragment,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # (resume logic BEFORE accelerate.prepare so states load correctly)
    epochs, losses, accuracies = [0], [0], [0]
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs = checkpoint["epochs"]
        losses = checkpoint["losses"]
        accuracies = checkpoint["accuracies"]
        args.start_epoch = epochs[-1]
        print(f"Resumed at epoch {epochs[-1]}")

    # ── wrap objects for multi‑GPU / mixed precision ─────────────────
    (
        data_loader_train,
        data_loader_val,
        model,
        optimizer,
    ) = accelerator.prepare(data_loader_train, data_loader_val, model, optimizer)  # ★ NEW

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ── training loop ────────────────────────────────────────────────
    print(f"Start training for {args.epochs} epochs 🔥")
    start_time = time.time()

    train_stats = training(
        model,
        data_loader_train,
        data_loader_val,
        accelerator,        # ★ NEW
        criterion,
        optimizer,
        scheduler,
        epochs,
        losses,
        accuracies,
        args=args,
    )

    if accelerator.is_main_process:
        print(
            f"Training on {len(dataset_train)} images, "
            f"{train_stats['epoch']} epochs\n"
            f"Last loss: {train_stats['loss']:.4f}  | "
            f"Fragment‑acc: {train_stats['accuracy']:.2f}%"
        )
        total_time = time.time() - start_time
        print("Training time:", str(datetime.timedelta(seconds=int(total_time))))
        
        # Log final results to wandb
        if not args.disable_wandb:
            wandb.log({
                "final/total_training_time": total_time,
                "final/final_loss": train_stats['loss'],
                "final/final_accuracy": train_stats['accuracy'],
                "final/total_epochs": train_stats['epoch']
            })
            wandb.finish()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    if args.smoke_test:          # ★ NEW
        args.epochs = 1          # ★ NEW
        args.batch_size = 8      # ★ NEW

    if args.output_dir and Accelerator().is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
