# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import os
import PIL
from PIL import Image

from torchvision import datasets, transforms

# Fix PIL decompression bomb error
Image.MAX_IMAGE_PIXELS = None

class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder with safe loading for large images"""
    
    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
            # Resize very large images
            max_size = (2048, 2048)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                print(f"Resizing image {path} from {img.size} to {max_size}")
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img


def build_dataset(is_train, args):
    transform = build_transform(args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = SafeImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(args):
    if args.size_puzzle == 225:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.Pad(padding=(0, 0, 1, 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
    if args.size_puzzle == 224:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
