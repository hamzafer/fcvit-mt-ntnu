# References:
# DeiT  : https://github.com/facebookresearch/deit
# MAE   : https://github.com/facebookresearch/mae
# --------------------------------------------------------

from tqdm import tqdm
import torch
import io, os
from accelerate import Accelerator


# ----------------------------------------------------------------------
def training(
    model,
    data_loader_train,
    data_loader_val,
    accelerator: Accelerator,         # ★ NEW
    criterion,
    optimizer,
    scheduler,
    epochs,
    losses,
    accuracies,
    args=None,
):
    """
    One full training loop over all epochs.

    All objects are already wrapped by `accelerator.prepare()` in main_train.py.
    """
    model.train(True)
    print_freq = 10

    for epoch in range(args.start_epoch, args.epochs):
        if accelerator.is_main_process:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}  |  learning rate: {lr:.2e}")

        running_loss = 0.0

        for batch_idx, (inputs, _) in tqdm(
            enumerate(data_loader_train, 0),
            total=len(data_loader_train),
            disable=not accelerator.is_local_main_process,
        ):
            optimizer.zero_grad()

            outputs, labels = model(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)           # ★ NEW
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % print_freq == 0 and accelerator.is_main_process:
                avg = running_loss / print_freq
                print(f"[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {avg:.4f}")
                epochs.append(epoch + 1)
                losses.append(avg)
                running_loss = 0.0

        scheduler.step()

        # save only every N epochs or at the last epoch
        N = 20  # Change this to your preferred interval
        if (
            args.output_dir
            and accelerator.is_main_process
            and ((epoch + 1) % N == 0 or (epoch + 1) == args.epochs)
        ):
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=epochs,
                losses=losses,
                accuracies=accuracies,
                args=args,
            )

        accuracies = val_model(
            model=model,
            data_loader_val=data_loader_val,
            accelerator=accelerator,            # ★ NEW
            accuracies=accuracies,
            epoch=epoch,
        )

    return {
        "epoch": epochs[-1],
        "loss": losses[-1],
        "accuracy": accuracies[-1],
    }


# ----------------------------------------------------------------------
def save_model(model, optimizer, scheduler, epochs, losses, accuracies, args):
    """Save a checkpoint at the *current* epoch (epochs[-1])."""
    chkpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epochs": epochs,
        "losses": losses,
        "accuracies": accuracies,
    }

    backbone = args.backbone.split("_")[1]
    grid     = int(args.num_fragment ** 0.5)
    curr_ep  = epochs[-1]

    fname = (
        f"FCViT_{backbone}_{grid}x{grid}"
        f"_ep{curr_ep:03d}_lr{args.lr:.0e}_b{args.batch_size}.pt"
    )
    fpath = os.path.join(args.output_dir, fname)

    torch.save(chkpt, fpath)
    # ---- logging -----------------------------------------------------
    size_mb = os.path.getsize(fpath) / (1024 ** 2)
    print(f"[🪄 save_model] Epoch {curr_ep:03d}  |  checkpoint → {fpath}  "
          f"({size_mb:.1f} MB)")

    # force write‑back to disk
    with open(fpath, "ab") as _f:
        os.fsync(_f.fileno())


# ----------------------------------------------------------------------
@torch.no_grad()
def val_model(model, data_loader_val, accelerator: Accelerator, accuracies, epoch=-1):
    model.eval()
    base_model = accelerator.unwrap_model(model)   # ★ NEW

    total, correct, correct_puzzle, num_fragment = 0, 0, 0, 0

    for _, (inputs, _) in tqdm(
        enumerate(data_loader_val, 0),
        total=len(data_loader_val),
        disable=not accelerator.is_local_main_process,
    ):
        outputs, labels = model(inputs)

        pred = outputs
        num_fragment = labels.size(1)
        total += labels.size(0)

        pred_   = base_model.mapping(pred)   # ★ CHANGED
        labels_ = base_model.mapping(labels) # ★ CHANGED
        correct += (pred_ == labels_).all(dim=2).sum().item()
        correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()

    # ---- gather stats across all processes --------------------------------
    total_tensor        = torch.tensor(total,          device=accelerator.device)
    correct_tensor      = torch.tensor(correct,        device=accelerator.device)
    correct_puzzle_tensor = torch.tensor(correct_puzzle, device=accelerator.device)

    gathered_total   = accelerator.gather(total_tensor).sum()
    gathered_corr    = accelerator.gather(correct_tensor).sum()
    gathered_corr_p  = accelerator.gather(correct_puzzle_tensor).sum()

    acc_fragment = 100 * gathered_corr / (gathered_total * num_fragment)
    acc_puzzle   = 100 * gathered_corr_p / gathered_total

    if accelerator.is_main_process:
        print(
            f"[Epoch {epoch + 1}] Fragment‑acc: {acc_fragment:.2f}%   "
            f"Puzzle‑acc: {acc_puzzle:.2f}%"
        )

    accuracies.append(acc_fragment.item())
    model.train(True)   # switch back to train mode
    return accuracies
