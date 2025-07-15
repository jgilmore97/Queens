import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # single‐logit forward
        logits = model(batch.x, batch.edge_index)            # [N]
        loss   = criterion(logits, batch.y.float())          # cast labels to float
        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

    return total_loss / total_nodes

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total_nodes = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index)            # [N]
        loss   = criterion(logits, batch.y.float())
        total_loss  += loss.item() * batch.num_nodes

        preds = (logits > 0).long()                          # threshold at 0
        correct    += (preds == batch.y).sum().item()
        total_nodes += batch.num_nodes

    return total_loss / total_nodes, correct / total_nodes

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = model.to(device)

    # compute positive‐class weight to counter class imbalance
    counter = Counter()
    for batch in train_loader:
        counter.update(batch.y.cpu().tolist())
    num_pos, num_neg = counter[1], counter[0]
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss:   {val_loss:.4f} | "
              f"Val Acc:    {val_acc*100:5.2f}% | "
              f"LR: {current_lr:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc*100:.2f}%")

class FocalLoss(nn.Module):
    """
    Binary focal loss for logits.
    Args
    ----
    alpha : float in [0,1]       — class-balancing factor (weight for positives)
    gamma : float ≥ 0            — focusing parameter (γ=2 is common)
    reduction : 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : shape [N]  (raw scores, BEFORE sigmoid)
        targets : shape [N]  (float 0/1)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # Convert BCE to pt (=sigmoid probs)
        pt  = torch.exp(-bce)          # pt = σ(logits) for y=1, 1-σ for y=0
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * bce
        if   self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum":  return loss.sum()
        else:                          return loss

