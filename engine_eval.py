# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224 \
#     --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 256 \
#     --data_path /cluster/home/muhamhz/data/imagenet \
#     --resume /cluster/home/muhamhz/fcvit-mt-ntnu/checkpoint/FCViT_base_3x3_ep100_lr3e-05_b64.pt
# --------------------------------------------------------

import os
import math
from tqdm import tqdm
import torch
from PIL import Image


def simple_map(coords, grid_size=3):
    """
    coords: Tensor of shape (..., 2), with integer row, col in [0..grid_size-1].
    Returns a LongTensor of shape (...) with values in 1..grid_size**2,
    row-major: (0,0)->1, (0,1)->2, (1,0)->4, etc.
    """
    return (coords[..., 0].long() * grid_size + coords[..., 1].long() + 1)

def save_visualization(img_path, labels_simple, pred_simple, output_root):
    """
    Given an image path and two orderings (lists of 1..N),
    splits the original image into a grid and reassembles:
      - original
      - shuffled (labels_simple)
      - predicted (pred_simple)
    Saves under output_root/<basename>/{original,shuffled,predicted}.jpg
    """
    img_name = os.path.basename(img_path)
    base, _ = os.path.splitext(img_name)
    out_dir = os.path.join(output_root, base)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(img_path).convert('RGB')
    grid_size = int(math.sqrt(len(labels_simple)))
    W, H = img.size
    patch_w, patch_h = W // grid_size, H // grid_size
    img = img.resize((patch_w * grid_size, patch_h * grid_size))

    # extract patches in row-major order
    patches = []
    for r in range(grid_size):
        for c in range(grid_size):
            left, upper = c * patch_w, r * patch_h
            patches.append(img.crop((left, upper, left + patch_w, upper + patch_h)))

    def assemble(order):
        canvas = Image.new('RGB', (patch_w * grid_size, patch_h * grid_size))
        for idx, pid in enumerate(order):
            r, c = divmod(idx, grid_size)
            # pid is 1-based
            canvas.paste(patches[pid - 1], (c * patch_w, r * patch_h))
        return canvas

    orig_img  = assemble(list(range(1, grid_size**2 + 1)))
    shuf_img  = assemble(labels_simple)
    pred_img  = assemble(pred_simple)

    orig_img.save(os.path.join(out_dir, 'original.jpg'))
    shuf_img.save(os.path.join(out_dir, 'shuffled.jpg'))
    pred_img.save(os.path.join(out_dir, 'predicted.jpg'))

@torch.no_grad()
def evaluate(data_loader, model, device, verbose=True, output_root='results'):
    model.eval()
    os.makedirs(output_root, exist_ok=True)

    total = correct = correct_puzzle = 0
    num_fragment = 0
    file_list = getattr(data_loader.dataset, 'samples', None) \
                or getattr(data_loader.dataset, 'imgs', None)

    with torch.no_grad():
        for batch_idx, (inputs, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = inputs.to(device)
            outputs, labels = model(inputs)

            pred = outputs
            num_fragment = labels.size(1)
            pred_   = model.mapping(pred)
            labels_ = model.mapping(labels)

            batch_size = labels_.size(0)
            total += batch_size
            correct += (pred_ == labels_).all(dim=2).sum().item()
            correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()

            grid_size = int(math.sqrt(num_fragment))
            pred_simple   = simple_map(pred_,   grid_size).cpu().tolist()
            labels_simple = simple_map(labels_, grid_size).cpu().tolist()

            if verbose and batch_idx < 3:
                print(f"\n----- Batch {batch_idx} -----")
                for i in range(min(2, batch_size)):
                    if file_list is not None:
                        idx = batch_idx * data_loader.batch_size + i
                        img_path = file_list[idx][0]
                        print(f"  Image: {img_path}")
                        save_visualization(img_path,
                                           labels_simple[i],
                                           pred_simple[i],
                                           output_root)

                    orig = labels_simple[i]
                    pr   = pred_simple[i]
                    match = [p == l for p, l in zip(pr, orig)]
                    print(f"Sample {i}:")
                    print(f"  Original order (1–9): {orig}")
                    print(f"  Predicted order (1–9): {pr}")
                    print(f"  Correct fragments:     {match}")
                    print(f"  All fragments correct: {all(match)}")

            # running accuracies
            running_frag = 100 * correct / (total * num_fragment)
            running_puz  = 100 * correct_puzzle / total
            print(f"After batch {batch_idx}: Frag acc {running_frag:.2f}%, "
                  f"Puz acc {running_puz:.2f}%")

    # final
    frag_acc = 100 * correct / (total * num_fragment)
    puz_acc  = 100 * correct_puzzle / total
    print("\n===== Evaluation Results =====")
    print(f"Fragment Accuracy: {frag_acc:.2f}%")
    print(f"Puzzle Accuracy:   {puz_acc:.2f}%")
    print(f"Total samples evaluated: {total}")
    print("==============================\n")

    return {'acc_fragment': frag_acc, 'acc_puzzle': puz_acc}
