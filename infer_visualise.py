import torch, random, pathlib
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from puzzle_fcvit import FCViT

# --- paths -------------------------------------------------------------
ckpt_path = "/cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_h100_normal/FCViT_base_3x3_ep100_lr3e-05_b64.pt"
img_root  = "/cluster/home/muhamhz/data/imagenet/val"
device    = "cuda" if torch.cuda.is_available() else "cpu"

# --- model -------------------------------------------------------------
ckpt   = torch.load(ckpt_path, map_location="cpu")
state  = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}  # strip DDP prefix
model  = FCViT(backbone="vit_base_patch16_224", num_fragment=9, size_fragment=75).to(device)
model.load_state_dict(state, strict=True)
model.eval()
model.augment_fragment = transforms.Resize((75, 75), antialias=True)

# --- image transforms --------------------------------------------------
tfs = transforms.Compose([
    transforms.Resize((225, 225), antialias=True),
    transforms.ToTensor()
])

# --- sample images -----------------------------------------------------
N = 4
img_paths = random.sample(list(pathlib.Path(img_root).rglob("*.JPEG")), N)
imgs = torch.stack([tfs(Image.open(p).convert("RGB")) for p in img_paths]).to(device)

# --- inference ---------------------------------------------------------
with torch.no_grad():
    pred, target = model(imgs)

pred_   = model.mapping(pred.clone())
target_ = model.mapping(target.clone())

acc_frag = (pred_ == target_).all(dim=2).float().mean().item() * 100
acc_puzz = (pred_ == target_).all(dim=2).all(dim=1).float().mean().item() * 100
print(f"Fragment‑level acc: {acc_frag:.2f}%")
print(f"Puzzle‑level   acc: {acc_puzz:.2f}%")

# --- visualisation -----------------------------------------------------
def unshuffle(tensor, order):
    C, H, W = tensor.shape
    p = 75
    pieces = [tensor[:, i:i+p, j:j+p] for i in range(0, H, p) for j in range(0, W, p)]
    grid   = [pieces[idx] for idx in order]
    rows   = [torch.cat(grid[i:i+3], dim=2) for i in range(0, 9, 3)]
    return torch.cat(rows, dim=1)

plt.figure(figsize=(12, 8))
for i, (img, ord_pred, ord_gt) in enumerate(zip(imgs.cpu(), pred_.cpu(), target_.cpu())):
    order_pred = torch.argmax((ord_pred[:, None, :] == model.map_coord).all(-1), dim=1)
    order_gt   = torch.argmax((ord_gt  [:, None, :] == model.map_coord).all(-1), dim=1)

    plt.subplot(N, 3, 3*i+1); plt.imshow(img.permute(1, 2, 0)); plt.axis('off'); plt.title("Original")
    plt.subplot(N, 3, 3*i+2); plt.imshow(unshuffle(img, order_gt).permute(1, 2, 0)); plt.axis('off'); plt.title("Shuffled input")
    plt.subplot(N, 3, 3*i+3); plt.imshow(unshuffle(img, order_pred).permute(1, 2, 0)); plt.axis('off'); plt.title("Reconstruction")

plt.tight_layout()
plt.show()
