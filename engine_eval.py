# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224 \
#     --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 256 \
#     --data_path /cluster/home/muhamhz/data/imagenet \
#     --resume /cluster/home/muhamhz/fcvit-mt-ntnu/checkpoint/FCViT_base_3x3_ep100_lr3e-05_b64.pt
# --------------------------------------------------------

import math
from tqdm import tqdm
import torch

def simple_map(coords, grid_size=3):
    """
    coords: Tensor of shape (..., 2), with integer row, col in [0..grid_size-1].
    Returns a LongTensor of shape (...) with values in 1..grid_size**2,
    row-major: (0,0)->1, (0,1)->2, (1,0)->4, etc.
    """
    return (coords[..., 0].long() * grid_size + coords[..., 1].long() + 1)

@torch.no_grad()
def evaluate(data_loader, model, device, verbose=True):
    model.eval()

    total = 0
    correct = 0
    correct_puzzle = 0
    num_fragment = 0

    with torch.no_grad():
        for batch_idx, (inputs, _) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            inputs = inputs.to(device)
            outputs, labels = model(inputs)

            pred = outputs
            num_fragment = labels.size(1)

            # get original (row, col) coords
            pred_   = model.mapping(pred)   # [B, num_fragment, 2]
            labels_ = model.mapping(labels) # [B, num_fragment, 2]

            # update metrics
            batch_size = labels_.size(0)
            total += batch_size
            correct += (pred_ == labels_).all(dim=2).sum().item()
            correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()

            # map to 1–9 indices
            grid_size     = int(math.sqrt(num_fragment))
            pred_simple   = simple_map(pred_,   grid_size)  # [B, num_fragment]
            labels_simple = simple_map(labels_, grid_size)  # [B, num_fragment]

            # print details for first few batches
            if verbose and batch_idx < 3:
                print(f"\n----- Batch {batch_idx} -----")
                for i in range(min(2, batch_size)):
                    orig  = labels_simple[i].cpu().tolist()
                    pr    = pred_simple[i].cpu().tolist()
                    match = [p == l for p, l in zip(pr, orig)]
                    print(f"Sample {i}:")
                    print(f"  Original order (1–9): {orig}")
                    print(f"  Predicted order (1–9): {pr}")
                    print(f"  Correct fragments:     {match}")
                    print(f"  All fragments correct: {all(match)}")

            # print running accuracies *every* batch
            running_frag_acc   = 100 * correct / (total * num_fragment)
            running_puzzle_acc = 100 * correct_puzzle / total
            print(f"After batch {batch_idx}: Running fragment accuracy: {running_frag_acc:.2f}%, "
                  f"puzzle accuracy: {running_puzzle_acc:.2f}%")

    # final results
    acc_fragment = 100 * correct / (total * num_fragment)
    acc_puzzle   = 100 * correct_puzzle / total

    print("\n===== Evaluation Results =====")
    print(f"Fragment Accuracy: {acc_fragment:.2f}%")
    print(f"Puzzle Accuracy:   {acc_puzzle:.2f}%")
    print(f"Total samples evaluated: {total}")
    print("==============================\n")

    return {'acc_fragment': acc_fragment, 'acc_puzzle': acc_puzzle}
