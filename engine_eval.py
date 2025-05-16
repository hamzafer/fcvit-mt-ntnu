# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# python main_eval.py --eval --device cuda:1 --backbone vit_base_patch16_224 --size_puzzle 225 --size_fragment 75 --num_fragment 9 --batch_size 256 --data_path /cluster/home/muhamhz/data/imagenet --resume /cluster/home/muhamhz/fcvit-mt-ntnu/checkpoint/FCViT_base_3x3_ep100_lr3e-05_b64.pt
# --------------------------------------------------------


from tqdm import tqdm

import torch


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
            total += labels.size(0)
            pred_ = model.mapping(pred)
            labels_ = model.mapping(labels)
            correct += (pred_ == labels_).all(dim=2).sum().item()
            correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()
            
            # Print sample results for the first few batches
            if verbose and batch_idx < 3:
                print(f"\n----- Batch {batch_idx} -----")
                # Show results for up to 2 samples in the batch
                for i in range(min(2, labels.size(0))):
                    print(f"Sample {i}:")
                    print(f"  Original order: {labels_[i].cpu().tolist()}")
                    print(f"  Predicted order: {pred_[i].cpu().tolist()}")
                    match = (pred_[i] == labels_[i]).all(dim=1).cpu().tolist()
                    print(f"  Correct fragments: {match}")
                    print(f"  All fragments correct: {all(match)}")

    acc_fragment = 100 * correct / (total * num_fragment)
    acc_puzzle = 100 * correct_puzzle / (total)
    
    # Print final results
    print("\n===== Evaluation Results =====")
    print(f"Fragment Accuracy: {acc_fragment:.2f}%")
    print(f"Puzzle Accuracy: {acc_puzzle:.2f}%")
    print(f"Total samples evaluated: {total}")
    print("==============================\n")

    return {'acc_fragment': acc_fragment, 'acc_puzzle': acc_puzzle}
