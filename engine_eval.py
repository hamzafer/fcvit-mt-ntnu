# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import datetime
import math
import sys
import time
from collections import defaultdict, deque
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy
from tqdm import tqdm


@torch.no_grad()
def evaluate(data_loader, model, device):
    model.eval()

    total = 0
    correct = 0
    correct_puzzle = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            inputs = inputs.to(device)

            outputs, labels = model(inputs)

            pred = outputs
            total += labels.size(0)
            pred_ = model.mapping(pred)
            labels_ = model.mapping(labels)
            correct += (pred_ == labels_).all(dim=2).sum().item()
            correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()

    acc_fragment = 100 * correct / (total * labels.size(1))
    acc_puzzle = 100 * correct_puzzle / (total)

    return {'acc_fragment': acc_fragment, 'acc_puzzle': acc_puzzle}
