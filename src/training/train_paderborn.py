import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.data.paderborn_dataset import build_paderborn_from_folder
from src.models.lstm_fcn import LSTMFCNClassifier


def eval_loader(model, loader, device, num_classes: int):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            ys.append(y.cpu())
            ps.append(pred.cpu())

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    labels = list(range(num_classes))
    cm = confusion_matrix(y, p, labels=labels)
    acc = float((p == y).mean())

    recalls = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recalls.append(tp / (tp + fn + 1e-9))
    bal = float(np.mean(recalls))
    return acc, bal, cm


def parse_args():
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Train Paderborn supervised baseline with LSTM-FCN")
    parser.add_argument("--data_root", type=Path, default=root / "data" / "raw" / "Paderborn" / "archive-2")
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--normalize", choices=["per_file", "train_global"], default="per_file")
    parser.add_argument("--include_conditions", nargs="*", default=None)
    parser.add_argument("--balance_train", action="store_true", default=True)
    parser.add_argument("--no_balance_train", action="store_false", dest="balance_train")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = Path(__file__).resolve().parent.parent.parent
    train_loader, val_loader, test_loader, meta = build_paderborn_from_folder(
        data_dir=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        split=tuple(args.split),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_train=args.balance_train,
        normalize=args.normalize,
        include_conditions=args.include_conditions,
    )
    print("meta:", meta)

    xb, yb = next(iter(train_loader))
    print("batch x:", xb.shape, "batch y unique:", sorted(set(yb.tolist())))

    model = LSTMFCNClassifier(num_classes=3, in_channels=1, lstm_hidden=128, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(root / "checkpoints", exist_ok=True)
    os.makedirs(root / "logs", exist_ok=True)

    train_losses, val_accs, val_bals = [], [], []
    best_val_bal = -1.0
    best_path = root / "checkpoints" / "best_paderborn_lstm_fcn.pt"
    log_path = root / "logs" / "paderborn_train_log.csv"

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        train_loss = total_loss / max(total, 1)
        val_acc, val_bal, _ = eval_loader(model, val_loader, device, num_classes=3)

        print(f"Epoch {ep:02d}/{args.epochs} train_loss={train_loss:.4f} | val_acc={val_acc:.4f} val_bal={val_bal:.4f}")

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_bals.append(val_bal)

        if val_bal > best_val_bal:
            best_val_bal = val_bal
            torch.save(model.state_dict(), best_path)
            print("  ✓ saved best")

    pd.DataFrame({
        "epoch": list(range(1, args.epochs + 1)),
        "train_loss": train_losses,
        "val_acc": val_accs,
        "val_bal": val_bals,
    }).to_csv(log_path, index=False)
    print("Saved log to:", log_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, test_bal, test_cm = eval_loader(model, test_loader, device, num_classes=3)
    print(f"[TEST] acc={test_acc:.4f} bal_acc={test_bal:.4f}")
    print("test confusion (rows=true, cols=pred), labels [healthy=0, inner=1, outer=2]:\n", test_cm)


if __name__ == "__main__":
    train()
