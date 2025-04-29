#!/usr/bin/env python
# find_best_ckpt_verbose.py
#
# Usage:
#   python find_best_ckpt_verbose.py /path/to/dir1 /path/to/dir2 …

# python pick_best.py \
#   /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100 \
#   /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal


import sys, pathlib, torch, time

def get_last_acc(checkpoint_path):
    """Return last validation accuracy stored in the file (float)."""
    data = torch.load(checkpoint_path,
                      map_location='cpu',
                      weights_only=True)          # fast / safe load
    if "accuracies" in data and data["accuracies"]:
        return float(data["accuracies"][-1])
    raise KeyError(f"{checkpoint_path} has no 'accuracies' list")

def scan_dir(dir_path):
    ckpts = sorted(pathlib.Path(dir_path).glob("*.pt"))
    if not ckpts:
        print(f"[{dir_path}]   ✖  no .pt files found\n")
        return

    print(f"\n[{dir_path}]   scanning {len(ckpts)} checkpoints …")
    best_file, best_acc = None, -1.0
    start = time.time()

    for idx, f in enumerate(ckpts, 1):
        try:
            acc = get_last_acc(f)
        except Exception as e:
            print(f"  {idx:3}/{len(ckpts)}  {f.name:<40}  ERROR → {e}")
            continue

        status = "☆" if acc > best_acc else " "
        print(f"  {idx:3}/{len(ckpts)}  {f.name:<40}  acc={acc:6.2f}% {status}")

        if acc > best_acc:
            best_acc, best_file = acc, f

    dur = time.time() - start
    if best_file:
        print(f"--→ best in {dir_path}  {best_acc:6.2f}%   {best_file.name}")
    print(f"scanned in {dur:.1f}s\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Give one or more checkpoint directories as arguments.")
        sys.exit(1)

    for d in sys.argv[1:]:
        scan_dir(d)

# (/cluster/home/muhamhz/fcvit_env) [muhamhz@idun-login1 util]$ 
# python pick_best.py \
#   /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100 \
#   /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal

# [/cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100]   scanning 100 checkpoints …
#     1/100  FCViT_base_3x3_ep001_lr3e-05_b64.pt       acc=  0.00% ☆
#     2/100  FCViT_base_3x3_ep002_lr3e-05_b64.pt       acc= 11.11% ☆
#     3/100  FCViT_base_3x3_ep003_lr3e-05_b64.pt       acc= 11.12% ☆
#     4/100  FCViT_base_3x3_ep004_lr3e-05_b64.pt       acc= 11.13% ☆
#     5/100  FCViT_base_3x3_ep005_lr3e-05_b64.pt       acc= 11.18% ☆
#     6/100  FCViT_base_3x3_ep006_lr3e-05_b64.pt       acc= 11.27% ☆
#     7/100  FCViT_base_3x3_ep007_lr3e-05_b64.pt       acc= 11.42% ☆
#     8/100  FCViT_base_3x3_ep008_lr3e-05_b64.pt       acc= 11.58% ☆
#     9/100  FCViT_base_3x3_ep009_lr3e-05_b64.pt       acc= 11.85% ☆
#    10/100  FCViT_base_3x3_ep010_lr3e-05_b64.pt       acc= 11.88% ☆
#    11/100  FCViT_base_3x3_ep011_lr3e-05_b64.pt       acc= 11.85%  
#    12/100  FCViT_base_3x3_ep012_lr3e-05_b64.pt       acc= 12.23% ☆
#    13/100  FCViT_base_3x3_ep013_lr3e-05_b64.pt       acc= 12.67% ☆
#    14/100  FCViT_base_3x3_ep014_lr3e-05_b64.pt       acc= 12.37%  
#    15/100  FCViT_base_3x3_ep015_lr3e-05_b64.pt       acc= 12.66%  
#    16/100  FCViT_base_3x3_ep016_lr3e-05_b64.pt       acc= 12.73% ☆
#    17/100  FCViT_base_3x3_ep017_lr3e-05_b64.pt       acc= 13.11% ☆
#    18/100  FCViT_base_3x3_ep018_lr3e-05_b64.pt       acc= 13.23% ☆
#    19/100  FCViT_base_3x3_ep019_lr3e-05_b64.pt       acc= 13.32% ☆
#    20/100  FCViT_base_3x3_ep020_lr3e-05_b64.pt       acc= 13.65% ☆
#    21/100  FCViT_base_3x3_ep021_lr3e-05_b64.pt       acc= 13.86% ☆
#    22/100  FCViT_base_3x3_ep022_lr3e-05_b64.pt       acc= 14.32% ☆
#    23/100  FCViT_base_3x3_ep023_lr3e-05_b64.pt       acc= 14.96% ☆
#    24/100  FCViT_base_3x3_ep024_lr3e-05_b64.pt       acc= 16.19% ☆
#    25/100  FCViT_base_3x3_ep025_lr3e-05_b64.pt       acc= 17.89% ☆
#    26/100  FCViT_base_3x3_ep026_lr3e-05_b64.pt       acc= 19.56% ☆
#    27/100  FCViT_base_3x3_ep027_lr3e-05_b64.pt       acc= 21.10% ☆
#    28/100  FCViT_base_3x3_ep028_lr3e-05_b64.pt       acc= 23.47% ☆
#    29/100  FCViT_base_3x3_ep029_lr3e-05_b64.pt       acc= 27.13% ☆
#    30/100  FCViT_base_3x3_ep030_lr3e-05_b64.pt       acc= 29.60% ☆
#    31/100  FCViT_base_3x3_ep031_lr3e-05_b64.pt       acc= 32.53% ☆
#    32/100  FCViT_base_3x3_ep032_lr3e-05_b64.pt       acc= 34.36% ☆
#    33/100  FCViT_base_3x3_ep033_lr3e-05_b64.pt       acc= 37.64% ☆
#    34/100  FCViT_base_3x3_ep034_lr3e-05_b64.pt       acc= 40.23% ☆
#    35/100  FCViT_base_3x3_ep035_lr3e-05_b64.pt       acc= 42.90% ☆
#    36/100  FCViT_base_3x3_ep036_lr3e-05_b64.pt       acc= 44.70% ☆
#    37/100  FCViT_base_3x3_ep037_lr3e-05_b64.pt       acc= 46.49% ☆
#    38/100  FCViT_base_3x3_ep038_lr3e-05_b64.pt       acc= 47.85% ☆
#    39/100  FCViT_base_3x3_ep039_lr3e-05_b64.pt       acc= 49.94% ☆
#    40/100  FCViT_base_3x3_ep040_lr3e-05_b64.pt       acc= 51.89% ☆
#    41/100  FCViT_base_3x3_ep041_lr3e-05_b64.pt       acc= 53.34% ☆
#    42/100  FCViT_base_3x3_ep042_lr3e-05_b64.pt       acc= 54.04% ☆
#    43/100  FCViT_base_3x3_ep043_lr3e-05_b64.pt       acc= 55.60% ☆
#    44/100  FCViT_base_3x3_ep044_lr3e-05_b64.pt       acc= 56.66% ☆
#    45/100  FCViT_base_3x3_ep045_lr3e-05_b64.pt       acc= 57.76% ☆
#    46/100  FCViT_base_3x3_ep046_lr3e-05_b64.pt       acc= 59.18% ☆
#    47/100  FCViT_base_3x3_ep047_lr3e-05_b64.pt       acc= 59.48% ☆
#    48/100  FCViT_base_3x3_ep048_lr3e-05_b64.pt       acc= 60.76% ☆
#    49/100  FCViT_base_3x3_ep049_lr3e-05_b64.pt       acc= 60.93% ☆
#    50/100  FCViT_base_3x3_ep050_lr3e-05_b64.pt       acc= 62.78% ☆
#    51/100  FCViT_base_3x3_ep051_lr3e-05_b64.pt       acc= 63.18% ☆
#    52/100  FCViT_base_3x3_ep052_lr3e-05_b64.pt       acc= 63.81% ☆
#    53/100  FCViT_base_3x3_ep053_lr3e-05_b64.pt       acc= 64.76% ☆
#    54/100  FCViT_base_3x3_ep054_lr3e-05_b64.pt       acc= 64.69%  
#    55/100  FCViT_base_3x3_ep055_lr3e-05_b64.pt       acc= 65.34% ☆
#    56/100  FCViT_base_3x3_ep056_lr3e-05_b64.pt       acc= 66.02% ☆
#    57/100  FCViT_base_3x3_ep057_lr3e-05_b64.pt       acc= 66.53% ☆
#    58/100  FCViT_base_3x3_ep058_lr3e-05_b64.pt       acc= 67.58% ☆
#    59/100  FCViT_base_3x3_ep059_lr3e-05_b64.pt       acc= 67.87% ☆
#    60/100  FCViT_base_3x3_ep060_lr3e-05_b64.pt       acc= 68.03% ☆
#    61/100  FCViT_base_3x3_ep061_lr3e-05_b64.pt       acc= 68.95% ☆
#    62/100  FCViT_base_3x3_ep062_lr3e-05_b64.pt       acc= 68.94%  
#    63/100  FCViT_base_3x3_ep063_lr3e-05_b64.pt       acc= 69.64% ☆
#    64/100  FCViT_base_3x3_ep064_lr3e-05_b64.pt       acc= 69.88% ☆
#    65/100  FCViT_base_3x3_ep065_lr3e-05_b64.pt       acc= 70.28% ☆
#    66/100  FCViT_base_3x3_ep066_lr3e-05_b64.pt       acc= 70.91% ☆
#    67/100  FCViT_base_3x3_ep067_lr3e-05_b64.pt       acc= 71.11% ☆
#    68/100  FCViT_base_3x3_ep068_lr3e-05_b64.pt       acc= 71.20% ☆
#    69/100  FCViT_base_3x3_ep069_lr3e-05_b64.pt       acc= 71.50% ☆
#    70/100  FCViT_base_3x3_ep070_lr3e-05_b64.pt       acc= 72.03% ☆
#    71/100  FCViT_base_3x3_ep071_lr3e-05_b64.pt       acc= 72.46% ☆
#    72/100  FCViT_base_3x3_ep072_lr3e-05_b64.pt       acc= 72.76% ☆
#    73/100  FCViT_base_3x3_ep073_lr3e-05_b64.pt       acc= 73.22% ☆
#    74/100  FCViT_base_3x3_ep074_lr3e-05_b64.pt       acc= 73.27% ☆
#    75/100  FCViT_base_3x3_ep075_lr3e-05_b64.pt       acc= 73.48% ☆
#    76/100  FCViT_base_3x3_ep076_lr3e-05_b64.pt       acc= 73.62% ☆
#    77/100  FCViT_base_3x3_ep077_lr3e-05_b64.pt       acc= 73.71% ☆
#    78/100  FCViT_base_3x3_ep078_lr3e-05_b64.pt       acc= 73.92% ☆
#    79/100  FCViT_base_3x3_ep079_lr3e-05_b64.pt       acc= 73.97% ☆
#    80/100  FCViT_base_3x3_ep080_lr3e-05_b64.pt       acc= 74.24% ☆
#    81/100  FCViT_base_3x3_ep081_lr3e-05_b64.pt       acc= 74.38% ☆
#    82/100  FCViT_base_3x3_ep082_lr3e-05_b64.pt       acc= 74.49% ☆
#    83/100  FCViT_base_3x3_ep083_lr3e-05_b64.pt       acc= 74.95% ☆
#    84/100  FCViT_base_3x3_ep084_lr3e-05_b64.pt       acc= 74.96% ☆
#    85/100  FCViT_base_3x3_ep085_lr3e-05_b64.pt       acc= 75.03% ☆
#    86/100  FCViT_base_3x3_ep086_lr3e-05_b64.pt       acc= 75.20% ☆
#    87/100  FCViT_base_3x3_ep087_lr3e-05_b64.pt       acc= 75.32% ☆
#    88/100  FCViT_base_3x3_ep088_lr3e-05_b64.pt       acc= 75.41% ☆
#    89/100  FCViT_base_3x3_ep089_lr3e-05_b64.pt       acc= 75.48% ☆
#    90/100  FCViT_base_3x3_ep090_lr3e-05_b64.pt       acc= 75.60% ☆
#    91/100  FCViT_base_3x3_ep091_lr3e-05_b64.pt       acc= 75.64% ☆
#    92/100  FCViT_base_3x3_ep092_lr3e-05_b64.pt       acc= 75.75% ☆
#    93/100  FCViT_base_3x3_ep093_lr3e-05_b64.pt       acc= 75.82% ☆
#    94/100  FCViT_base_3x3_ep094_lr3e-05_b64.pt       acc= 75.78%  
#    95/100  FCViT_base_3x3_ep095_lr3e-05_b64.pt       acc= 76.05% ☆
#    96/100  FCViT_base_3x3_ep096_lr3e-05_b64.pt       acc= 76.00%  
#    97/100  FCViT_base_3x3_ep097_lr3e-05_b64.pt       acc= 76.02%  
#    98/100  FCViT_base_3x3_ep098_lr3e-05_b64.pt       acc= 76.02%  
#    99/100  FCViT_base_3x3_ep099_lr3e-05_b64.pt       acc= 76.03%  
#   100/100  FCViT_base_3x3_ep100_lr3e-05_b64.pt       acc= 76.02%  
# --→ best in /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100   76.05%   FCViT_base_3x3_ep095_lr3e-05_b64.pt
# scanned in 288.5s


# [/cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal]   scanning 100 checkpoints …
#     1/100  FCViT_base_3x3_ep001_lr3e-05_b64.pt       acc=  0.00% ☆
#     2/100  FCViT_base_3x3_ep002_lr3e-05_b64.pt       acc= 11.11% ☆
#     3/100  FCViT_base_3x3_ep003_lr3e-05_b64.pt       acc= 11.12% ☆
#     4/100  FCViT_base_3x3_ep004_lr3e-05_b64.pt       acc= 11.13% ☆
#     5/100  FCViT_base_3x3_ep005_lr3e-05_b64.pt       acc= 11.18% ☆
#     6/100  FCViT_base_3x3_ep006_lr3e-05_b64.pt       acc= 11.27% ☆
#     7/100  FCViT_base_3x3_ep007_lr3e-05_b64.pt       acc= 11.42% ☆
#     8/100  FCViT_base_3x3_ep008_lr3e-05_b64.pt       acc= 11.58% ☆
#     9/100  FCViT_base_3x3_ep009_lr3e-05_b64.pt       acc= 11.85% ☆
#    10/100  FCViT_base_3x3_ep010_lr3e-05_b64.pt       acc= 11.88% ☆
#    11/100  FCViT_base_3x3_ep011_lr3e-05_b64.pt       acc= 11.85%  
#    12/100  FCViT_base_3x3_ep012_lr3e-05_b64.pt       acc= 12.23% ☆
#    13/100  FCViT_base_3x3_ep013_lr3e-05_b64.pt       acc= 12.67% ☆
#    14/100  FCViT_base_3x3_ep014_lr3e-05_b64.pt       acc= 12.37%  
#    15/100  FCViT_base_3x3_ep015_lr3e-05_b64.pt       acc= 12.66%  
#    16/100  FCViT_base_3x3_ep016_lr3e-05_b64.pt       acc= 12.73% ☆
#    17/100  FCViT_base_3x3_ep017_lr3e-05_b64.pt       acc= 13.11% ☆
#    18/100  FCViT_base_3x3_ep018_lr3e-05_b64.pt       acc= 13.23% ☆
#    19/100  FCViT_base_3x3_ep019_lr3e-05_b64.pt       acc= 13.32% ☆
#    20/100  FCViT_base_3x3_ep020_lr3e-05_b64.pt       acc= 13.65% ☆
#    21/100  FCViT_base_3x3_ep021_lr3e-05_b64.pt       acc= 13.86% ☆
#    22/100  FCViT_base_3x3_ep022_lr3e-05_b64.pt       acc= 14.32% ☆
#    23/100  FCViT_base_3x3_ep023_lr3e-05_b64.pt       acc= 14.96% ☆
#    24/100  FCViT_base_3x3_ep024_lr3e-05_b64.pt       acc= 16.19% ☆
#    25/100  FCViT_base_3x3_ep025_lr3e-05_b64.pt       acc= 17.89% ☆
#    26/100  FCViT_base_3x3_ep026_lr3e-05_b64.pt       acc= 19.56% ☆
#    27/100  FCViT_base_3x3_ep027_lr3e-05_b64.pt       acc= 21.10% ☆
#    28/100  FCViT_base_3x3_ep028_lr3e-05_b64.pt       acc= 23.47% ☆
#    29/100  FCViT_base_3x3_ep029_lr3e-05_b64.pt       acc= 27.13% ☆
#    30/100  FCViT_base_3x3_ep030_lr3e-05_b64.pt       acc= 29.60% ☆
#    31/100  FCViT_base_3x3_ep031_lr3e-05_b64.pt       acc= 32.53% ☆
#    32/100  FCViT_base_3x3_ep032_lr3e-05_b64.pt       acc= 34.36% ☆
#    33/100  FCViT_base_3x3_ep033_lr3e-05_b64.pt       acc= 37.64% ☆
#    34/100  FCViT_base_3x3_ep034_lr3e-05_b64.pt       acc= 40.23% ☆
#    35/100  FCViT_base_3x3_ep035_lr3e-05_b64.pt       acc= 42.90% ☆
#    36/100  FCViT_base_3x3_ep036_lr3e-05_b64.pt       acc= 44.70% ☆
#    37/100  FCViT_base_3x3_ep037_lr3e-05_b64.pt       acc= 46.49% ☆
#    38/100  FCViT_base_3x3_ep038_lr3e-05_b64.pt       acc= 47.85% ☆
#    39/100  FCViT_base_3x3_ep039_lr3e-05_b64.pt       acc= 49.94% ☆
#    40/100  FCViT_base_3x3_ep040_lr3e-05_b64.pt       acc= 51.89% ☆
#    41/100  FCViT_base_3x3_ep041_lr3e-05_b64.pt       acc= 53.34% ☆
#    42/100  FCViT_base_3x3_ep042_lr3e-05_b64.pt       acc= 54.04% ☆
#    43/100  FCViT_base_3x3_ep043_lr3e-05_b64.pt       acc= 55.60% ☆
#    44/100  FCViT_base_3x3_ep044_lr3e-05_b64.pt       acc= 56.66% ☆
#    45/100  FCViT_base_3x3_ep045_lr3e-05_b64.pt       acc= 57.76% ☆
#    46/100  FCViT_base_3x3_ep046_lr3e-05_b64.pt       acc= 59.18% ☆
#    47/100  FCViT_base_3x3_ep047_lr3e-05_b64.pt       acc= 59.48% ☆
#    48/100  FCViT_base_3x3_ep048_lr3e-05_b64.pt       acc= 60.76% ☆
#    49/100  FCViT_base_3x3_ep049_lr3e-05_b64.pt       acc= 60.93% ☆
#    50/100  FCViT_base_3x3_ep050_lr3e-05_b64.pt       acc= 62.78% ☆
#    51/100  FCViT_base_3x3_ep051_lr3e-05_b64.pt       acc= 63.18% ☆
#    52/100  FCViT_base_3x3_ep052_lr3e-05_b64.pt       acc= 63.81% ☆
#    53/100  FCViT_base_3x3_ep053_lr3e-05_b64.pt       acc= 64.76% ☆
#    54/100  FCViT_base_3x3_ep054_lr3e-05_b64.pt       acc= 64.69%  
#    55/100  FCViT_base_3x3_ep055_lr3e-05_b64.pt       acc= 65.34% ☆
#    56/100  FCViT_base_3x3_ep056_lr3e-05_b64.pt       acc= 66.02% ☆
#    57/100  FCViT_base_3x3_ep057_lr3e-05_b64.pt       acc= 66.53% ☆
#    58/100  FCViT_base_3x3_ep058_lr3e-05_b64.pt       acc= 67.58% ☆
#    59/100  FCViT_base_3x3_ep059_lr3e-05_b64.pt       acc= 67.87% ☆
#    60/100  FCViT_base_3x3_ep060_lr3e-05_b64.pt       acc= 68.03% ☆
#    61/100  FCViT_base_3x3_ep061_lr3e-05_b64.pt       acc= 68.95% ☆

#    62/100  FCViT_base_3x3_ep062_lr3e-05_b64.pt       acc= 68.94%  
#    63/100  FCViT_base_3x3_ep063_lr3e-05_b64.pt       acc= 69.64% ☆
#    64/100  FCViT_base_3x3_ep064_lr3e-05_b64.pt       acc= 69.88% ☆
#    65/100  FCViT_base_3x3_ep065_lr3e-05_b64.pt       acc= 70.28% ☆
#    66/100  FCViT_base_3x3_ep066_lr3e-05_b64.pt       acc= 70.91% ☆
#    67/100  FCViT_base_3x3_ep067_lr3e-05_b64.pt       acc= 71.11% ☆
#    68/100  FCViT_base_3x3_ep068_lr3e-05_b64.pt       acc= 71.20% ☆
#    69/100  FCViT_base_3x3_ep069_lr3e-05_b64.pt       acc= 71.50% ☆
#    70/100  FCViT_base_3x3_ep070_lr3e-05_b64.pt       acc= 72.03% ☆
#    71/100  FCViT_base_3x3_ep071_lr3e-05_b64.pt       acc= 72.46% ☆
#    72/100  FCViT_base_3x3_ep072_lr3e-05_b64.pt       acc= 72.76% ☆
#    73/100  FCViT_base_3x3_ep073_lr3e-05_b64.pt       acc= 73.22% ☆
#    74/100  FCViT_base_3x3_ep074_lr3e-05_b64.pt       acc= 73.27% ☆
#    75/100  FCViT_base_3x3_ep075_lr3e-05_b64.pt       acc= 73.48% ☆
#    76/100  FCViT_base_3x3_ep076_lr3e-05_b64.pt       acc= 73.62% ☆
#    77/100  FCViT_base_3x3_ep077_lr3e-05_b64.pt       acc= 73.71% ☆
#    78/100  FCViT_base_3x3_ep078_lr3e-05_b64.pt       acc= 73.92% ☆
#    79/100  FCViT_base_3x3_ep079_lr3e-05_b64.pt       acc= 73.97% ☆
#    80/100  FCViT_base_3x3_ep080_lr3e-05_b64.pt       acc= 74.24% ☆
#    81/100  FCViT_base_3x3_ep081_lr3e-05_b64.pt       acc= 74.38% ☆
#    82/100  FCViT_base_3x3_ep082_lr3e-05_b64.pt       acc= 74.49% ☆
#    83/100  FCViT_base_3x3_ep083_lr3e-05_b64.pt       acc= 74.95% ☆
#    84/100  FCViT_base_3x3_ep084_lr3e-05_b64.pt       acc= 74.96% ☆
#    85/100  FCViT_base_3x3_ep085_lr3e-05_b64.pt       acc= 75.03% ☆
#    86/100  FCViT_base_3x3_ep086_lr3e-05_b64.pt       acc= 75.20% ☆
#    87/100  FCViT_base_3x3_ep087_lr3e-05_b64.pt       acc= 75.32% ☆
#    88/100  FCViT_base_3x3_ep088_lr3e-05_b64.pt       acc= 75.41% ☆
#    89/100  FCViT_base_3x3_ep089_lr3e-05_b64.pt       acc= 75.48% ☆
#    90/100  FCViT_base_3x3_ep090_lr3e-05_b64.pt       acc= 75.60% ☆
#    91/100  FCViT_base_3x3_ep091_lr3e-05_b64.pt       acc= 75.64% ☆
#    92/100  FCViT_base_3x3_ep092_lr3e-05_b64.pt       acc= 75.75% ☆
#    93/100  FCViT_base_3x3_ep093_lr3e-05_b64.pt       acc= 75.82% ☆
#    94/100  FCViT_base_3x3_ep094_lr3e-05_b64.pt       acc= 75.78%  
#    95/100  FCViT_base_3x3_ep095_lr3e-05_b64.pt       acc= 76.05% ☆
#    96/100  FCViT_base_3x3_ep096_lr3e-05_b64.pt       acc= 76.00%  
#    97/100  FCViT_base_3x3_ep097_lr3e-05_b64.pt       acc= 76.02%  
#    98/100  FCViT_base_3x3_ep098_lr3e-05_b64.pt       acc= 76.02%  
#    99/100  FCViT_base_3x3_ep099_lr3e-05_b64.pt       acc= 76.03%  
#   100/100  FCViT_base_3x3_ep100_lr3e-05_b64.pt       acc= 76.02%  
# --→ best in /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal   76.05%   FCViT_base_3x3_ep095_lr3e-05_b64.pt
# scanned in 459.3s
