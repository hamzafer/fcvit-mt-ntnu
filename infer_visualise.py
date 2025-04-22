# infer_visualise_simple.py
import torch, random, pathlib
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from puzzle_fcvit import FCViT
from datetime import datetime

# ---------- paths ----------
ckpt_path = "/cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal/FCViT_base_3x3_ep100_lr3e-05_b64.pt"
img_root  = "/cluster/home/muhamhz/data/imagenet/val"
device    = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------- load model ----------
ckpt  = torch.load(ckpt_path, map_location="cpu")
state = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}  # strip DDP prefix

model = FCViT(backbone="vit_base_patch16_224",
              num_fragment=9,
              size_fragment=75).to(device)
model.load_state_dict(state, strict=True)
model.eval()
model.augment_fragment = transforms.Resize((75, 75), antialias=True)
print("Checkpoint loaded")

# ---------- sample images ----------
N = 4
img_paths = random.sample(list(pathlib.Path(img_root).rglob("*.JPEG")), N)
print("Using images:")
for p in img_paths:
    print(" •", p)

tfs = transforms.Compose([
    transforms.Resize((225, 225), antialias=True),
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
print(f"Fragment‑level accuracy: {acc_frag:.2f}%")
print(f"Puzzle‑level accuracy  : {acc_puzz:.2f}%")

# ---------- prepare for plotting ----------
imgs_cpu  = imgs_gpu.cpu()
pred_cpu  = pred_.cpu()
tgt_cpu   = tgt_.cpu()
map_coord = model.map_coord.cpu()

def unshuffle(tensor, order):
    C, H, W = tensor.shape
    p = 75
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

    # column 1 – ORIGINAL (ground‑truth order)
    plt.subplot(N, 3, 3*i+1)
    plt.imshow(unshuffle(img, ord_gt_i).permute(1, 2, 0))
    plt.axis('off'); plt.title("Original")

    # column 2 – SHUFFLED input
    plt.subplot(N, 3, 3*i+2)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off'); plt.title("Shuffled")

    # column 3 – MODEL reconstruction
    plt.subplot(N, 3, 3*i+3)
    plt.imshow(unshuffle(img, ord_pred_i).permute(1, 2, 0))
    plt.axis('off'); plt.title("Reconstruction")

plt.tight_layout()

results_dir = pathlib.Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)
out_png = results_dir / f"recon_vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

plt.savefig(out_png, dpi=600)
print("Saved figure to", out_png)
plt.show()
