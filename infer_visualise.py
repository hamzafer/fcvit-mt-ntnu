#!/usr/bin/env python
# infer_visualise_log.py
#
# • saves a figure + a plain‑text log for each run
# • all tunable parameters are grouped at the top

import torch, random, pathlib, sys
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from datetime import datetime
from puzzle_fcvit import FCViT


# ────────────────────────────── CONFIG ────────────────────────────── #
CKPT_PATH   = "/cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal/FCViT_base_3x3_ep100_lr3e-05_b64.pt"
IMG_ROOT    = "/cluster/home/muhamhz/data/imagenet/val"
N_IMAGES    = 4                      # how many images to visualise
BATCH_SIZE  = N_IMAGES               # one pass – no dataloader
BACKBONE    = "vit_base_patch16_224"
PUZZLE_SIZE = 225
FRAG_SIZE   = 75
NUM_FRAG    = 9
# ──────────────────────────────────────────────────────────────────── #

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# results directory
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = pathlib.Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)
log_path  = results_dir / f"log_{STAMP}.txt"
fig_path  = results_dir / f"recon_vis_{STAMP}.png"

# simple file logger
log_file = open(log_path, "w")
def log(msg):
    print(msg)
    print(msg, file=log_file)
log(f"Timestamp         : {STAMP}")
log(f"Checkpoint        : {CKPT_PATH}")
log(f"Image root        : {IMG_ROOT}")
log(f"Device            : {device}")

# ---------- load model ----------
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
state = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}  # strip DDP prefix

model = FCViT(backbone=BACKBONE, num_fragment=NUM_FRAG, size_fragment=FRAG_SIZE).to(device)
model.load_state_dict(state, strict=True)
model.eval()
model.augment_fragment = transforms.Resize((FRAG_SIZE, FRAG_SIZE), antialias=True)
log("Checkpoint loaded and model in eval mode")

# ---------- sample images ----------
img_paths = random.sample(list(pathlib.Path(IMG_ROOT).rglob("*.JPEG")), N_IMAGES)
log(f"Sampled {N_IMAGES} images:")
for p in img_paths:
    log(f" • {p}")

tfs = transforms.Compose([
    transforms.Resize((PUZZLE_SIZE, PUZZLE_SIZE), antialias=True),
    transforms.ToTensor()
])
imgs_gpu = torch.stack([tfs(Image.open(p).convert("RGB")) for p in img_paths]).to(device)

# ---------- inference ----------
with torch.no_grad():
    pred_gpu, tgt_gpu = model(imgs_gpu)

pred_ = model.mapping(pred_gpu.clone())
tgt_  = model.mapping(tgt_gpu.clone())

acc_frag = (pred_ == tgt_).all(dim=2).float().mean().item() * 100
acc_puzz = (pred_ == tgt_).all(dim=2).all(dim=1).float().mean().item() * 100
log(f"Fragment‑level accuracy : {acc_frag:.2f}%")
log(f"Puzzle‑level accuracy   : {acc_puzz:.2f}%")

# ---------- prepare for plotting ----------
imgs_cpu  = imgs_gpu.cpu()
pred_cpu  = pred_.cpu()
tgt_cpu   = tgt_.cpu()
map_coord = model.map_coord.cpu()

def unshuffle(tensor, order):
    C, H, W = tensor.shape
    p = FRAG_SIZE
    pieces = [tensor[:, i:i+p, j:j+p] for i in range(0, H, p) for j in range(0, W, p)]
    grid   = [pieces[idx] for idx in order]
    rows   = [torch.cat(grid[i:i+3], dim=2) for i in range(0, 9, 3)]
    return torch.cat(rows, dim=1)

# ---------- visualise ----------
plt.figure(figsize=(12, 8))
for i, (img, ord_pred, ord_gt) in enumerate(zip(imgs_cpu, pred_cpu, tgt_cpu)):
    mask_pred = (ord_pred[:, None, :] == map_coord).all(-1).long()
    mask_gt   = (ord_gt  [:, None, :] == map_coord).all(-1).long()
    ord_pred_i = mask_pred.argmax(dim=1)
    ord_gt_i   = mask_gt.argmax(dim=1)

    # column 1 – ORIGINAL
    plt.subplot(N_IMAGES, 3, 3*i+1)
    plt.imshow(unshuffle(img, ord_gt_i).permute(1, 2, 0))
    plt.axis('off'); plt.title("Original")

    # column 2 – SHUFFLED
    plt.subplot(N_IMAGES, 3, 3*i+2)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off'); plt.title("Shuffled")

    # column 3 – RECONSTRUCTION
    plt.subplot(N_IMAGES, 3, 3*i+3)
    plt.imshow(unshuffle(img, ord_pred_i).permute(1, 2, 0))
    plt.axis('off'); plt.title("Reconstruction")

plt.tight_layout()
plt.savefig(fig_path, dpi=600)
log(f"Figure saved to    : {fig_path}")
log(f"Log saved to       : {log_path}")
log_file.close()
print("Done.")
