# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import math
import sys
from typing import Iterable
from tqdm import tqdm

import torch


def training(model, data_loader_train, data_loader_val, device,
             criterion, optimizer, scheduler,
             epochs, losses, accuracies,
             args=None):
    model.train(True)
    print_freq = 10  # 100

    for epoch in range(args.start_epoch, args.epochs):
        print(f"epoch {epoch + 1} learning rate : {optimizer.param_groups[0]['lr']}")
        running_loss = 0.
        for batch_idx, (inputs, _) in tqdm(enumerate(data_loader_train, 0), total=len(data_loader_train)):
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs, labels = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % print_freq == print_freq - 1:
                print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {running_loss / print_freq:.4f}')
                epochs.append(epoch + 1)
                losses.append(running_loss / print_freq)
                running_loss = 0.
        scheduler.step()
        if args.output_dir:
            save_model(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epochs=epochs, losses=losses, accuracies=accuracies,
                args=args
            )
        accuracies = val_model(
            model=model, data_loader_val=data_loader_val, device=device,
            accuracies=accuracies, epoch=epoch
        )
    return {'epoch': epochs[-1], 'loss': losses[-1], 'accuracy': accuracies[-1]}


def save_model(model, optimizer, scheduler, epochs, losses, accuracies, args):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epochs': epochs,
        'losses': losses,
        'accuracies': accuracies,
    }
    backbone = args.backbone.split('_')[1]
    puzzle_type = f'{int(args.num_fragment ** 0.5)}x{int(args.num_fragment ** 0.5)}'
    model_path = args.output_dir + '/' + f'FCViT_{backbone}_{puzzle_type}_ep{args.epochs}_lr{args.lr:.0e}_b{args.batch_size}.pt'
    torch.save(checkpoint, model_path)
    print(f"****** Model checkpoint saved at epochs {epochs[-1]} ******")


def val_model(model, data_loader_val, device, accuracies, epoch=-1):
    model.eval()

    total = 0
    correct = 0
    correct_puzzle = 0
    num_fragment = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in tqdm(enumerate(data_loader_val, 0), total=len(data_loader_val)):
            inputs = inputs.to(device)

            outputs, labels = model(inputs)

            pred = outputs
            num_fragment = labels.size(1)
            total += labels.size(0)
            pred_ = model.mapping(pred)
            labels_ = model.mapping(labels)
            correct += (pred_ == labels_).all(dim=2).sum().item()
            correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()

    acc_fragment = 100 * correct / (total * num_fragment)
    acc_puzzle = 100 * correct_puzzle / (total)
    print(f'[Epoch {epoch + 1}] Accuracy (Fragment-level) on the test set: {acc_fragment:.2f}%')
    print(f'[Epoch {epoch + 1}] Accuracy (Puzzle-level) on the test set: {acc_puzzle:.2f}%')
    accuracies.append(acc_fragment)
    return accuracies
