import os

# Step 1: Mount your Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Example: Navigate to "My Drive/Projects/MyFolder"
target_folder = '/content/drive/My Drive/queens_project'

# Change the working directory
os.chdir(target_folder)

# Confirm you're in the right directory
print("Current working directory:", os.getcwd())

# --- 1) (Re)install & imports ---
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm

from train import *
from model import *
from data_loader import *

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter
from tqdm.auto import tqdm
import random

TRAINVAL_JSON = "10k_training_set_with_states.json"
TEST_JSON     = "test_set_with_states.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

train_loader, val_loader = get_queens_loaders(
    TRAINVAL_JSON,
    batch_size=512,
    val_ratio=0.10,
    seed=42,
    num_workers=4,
    pin_memory=True,
    shuffle_train=True,
)

test_dataset = QueensDataset(
    TEST_JSON,
    split="all",
    val_ratio=0.0,
    seed=42,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

def sample_pos_neg(dataset, device, frac=0.1, seed=42):
    """
    Randomly sample `frac` of dataset examples, count y=0 vs y=1, and return
    (num_pos, num_neg).

    dataset: torch_geometric Dataset with .__getitem__ returning a Data.y Tensor
    device:  torch device where we'll eventually put our alpha/gamma tensors
    frac:    fraction of the dataset to sample (0 < frac <= 1)
    seed:    random seed for reproducibility
    """
    N = len(dataset)
    k = max(1, int(frac * N))
    rng = random.Random(seed)
    sample_idx = rng.sample(range(N), k)

    counter = Counter()
    for idx in sample_idx:
        data = dataset[idx]
        # data.y is [num_nodes] of 0/1 labels
        counter.update(data.y.tolist())

    num_pos = counter[1]
    num_neg = counter[0]
    print(f"Sampled {k}/{N} graphs → {num_pos} positives, {num_neg} negatives.")
    return num_pos, num_neg

# Cell 4: Model, loss, optimizer, scheduler
model = GAT(input_dim=14, hidden_dim=256, layer_count=3, dropout=0.2, heads = 2).to(device)

dataset = train_loader.dataset  # or your QueensDataset instance
num_pos, num_neg = sample_pos_neg(dataset, device="cpu", frac=0.1, seed=123)
alpha     = num_pos / (num_pos + num_neg)   # weight positives ≈ prevalence
criterion = FocalLoss(alpha=alpha, gamma=2.0)   # γ=2 is a common default

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# ---------- Training & validation loop with full metrics ----------
num_epochs = 30
history = {
    "train_loss": [], "train_acc": [], "train_prec": [], "train_rec": [], "train_f1": [],
    "val_loss":   [], "val_acc":   [], "val_prec":   [], "val_rec":   [], "val_f1":   [],
    "grad_norm": [], "lr": []
}

eps = 1e-9                                           # to avoid 0-div in precision/recall

for epoch in range(1, num_epochs + 1):
    # ======== TRAIN ========
    model.train()
    tloss, tTP, tFP, tFN, tTN, grad_sum = 0.0, 0, 0, 0, 0, 0.0

    train_bar = tqdm(train_loader, desc=f"[{epoch}/{num_epochs}] Train", unit="batch", leave=False)
    for batch in train_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)            # [N]
        loss   = criterion(logits, batch.y.float())
        loss.backward()

        # grad-norm (per-mini-batch)
        grad_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in model.parameters() if p.grad is not None))
        grad_sum += grad_norm.item()

        optimizer.step()

        # confusion counts for this batch
        preds = (logits > 0).long()
        tTP += ((preds == 1) & (batch.y == 1)).sum().item()
        tFP += ((preds == 1) & (batch.y == 0)).sum().item()
        tFN += ((preds == 0) & (batch.y == 1)).sum().item()
        tTN += ((preds == 0) & (batch.y == 0)).sum().item()

        tloss += loss.item() * batch.num_nodes
        train_bar.set_postfix(loss=f"{tloss/(tTP+tFP+tFN+tTN):.4f}")

    train_loss = tloss / (tTP + tFP + tFN + tTN)
    train_prec = tTP / (tTP + tFP + eps)
    train_rec  = tTP / (tTP + tFN + eps)
    train_f1   = 2 * train_prec * train_rec / (train_prec + train_rec + eps)
    train_acc  = (tTP + tTN) / (tTP + tFP + tFN + tTN)
    history["grad_norm"].append(grad_sum / len(train_loader))

    # ======== VALIDATE ========
    model.eval()
    vloss, vTP, vFP, vFN, vTN = 0.0, 0, 0, 0, 0
    val_bar = tqdm(val_loader, desc=f"[{epoch}/{num_epochs}] Val  ", unit="batch", leave=False)

    with torch.no_grad():
        for batch in val_bar:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss   = criterion(logits, batch.y.float())
            preds  = (logits > 0).long()

            vTP += ((preds == 1) & (batch.y == 1)).sum().item()
            vFP += ((preds == 1) & (batch.y == 0)).sum().item()
            vFN += ((preds == 0) & (batch.y == 1)).sum().item()
            vTN += ((preds == 0) & (batch.y == 0)).sum().item()

            vloss += loss.item() * batch.num_nodes

    val_loss = vloss / (vTP + vFP + vFN + vTN)
    val_prec = vTP / (vTP + vFP + eps)
    val_rec  = vTP / (vTP + vFN + eps)
    val_f1   = 2 * val_prec * val_rec / (val_prec + val_rec + eps)
    val_acc  = (vTP + vTN) / (vTP + vFP + vFN + vTN)

    # ======== Scheduler & logging ========
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:02d} | "
        f"Train L {train_loss:.4f} P {train_prec:.3f} R {train_rec:.3f} F1 {train_f1:.3f} | "
        f"Val L {val_loss:.4f} P {val_prec:.3f} R {val_rec:.3f} F1 {val_f1:.3f} | "
        f"Acc {val_acc*100:5.2f}% | Grad {history['grad_norm'][-1]:.3f} | LR {lr_now:.1e}"
    )

    # store
    history["train_loss"].append(train_loss)
    history["train_prec"].append(train_prec)
    history["train_rec"].append(train_rec)
    history["train_f1"].append(train_f1)
    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_prec"].append(val_prec)
    history["val_rec"].append(val_rec)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)
    history["lr"].append(lr_now)

# Cell 6: Final test evaluation
model.eval()
test_loss, correct, test_nodes = 0.0, 0, 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        loss   = criterion(logits, batch.y.float())
        test_loss += loss.item() * batch.num_nodes
        preds     = (logits > 0).long()
        correct  += (preds == batch.y).sum().item()
        test_nodes += batch.num_nodes

test_loss /= test_nodes
test_acc   = correct / test_nodes

print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
