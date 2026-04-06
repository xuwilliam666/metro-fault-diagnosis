import os
from pathlib import Path
import argparse


def _sanitize_omp_num_threads():
    value = os.environ.get("OMP_NUM_THREADS")
    if value is None:
        return

    try:
        if int(value) <= 0:
            raise ValueError
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"
        print("[WARN] Invalid OMP_NUM_THREADS detected, reset to 1")


_sanitize_omp_num_threads()

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from src.data.cwru_dataset import build_cwru_from_folder
from src.models.lstm_fcn import LSTMFCNClassifier


def parse_args():
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Train CWRU supervised baseline with LSTM-FCN")
    parser.add_argument("--data_root", type=Path, default=root / "data" / "raw" / "CWRU" / "12k_DE")
    parser.add_argument("--label_mode", choices=["fault3", "health_inner_outer", "inner_outer"], default="fault3")
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--balance_train", action="store_true", default=True)
    parser.add_argument("--no_balance_train", action="store_false", dest="balance_train")
    parser.add_argument("--normalize", choices=["per_file", "train_global"], default="per_file")
    return parser.parse_args()


def build_label_map(label_mode: str):
    if label_mode == "inner_outer":
        return {
            "ir": 0,
            "or": 1,
        }

    if label_mode == "health_inner_outer":
        return {
            "normal": 0,
            "ir": 1,
            "or": 2,
        }

    return {
        "ir": 0,
        "or": 1,
        "b007": 2,
        "b014": 2,
        "b021": 2,
        "b028": 2,
    }


def label_mode_names(label_mode: str):
    if label_mode == "inner_outer":
        return ["inner=0", "outer=1"]
    if label_mode == "health_inner_outer":
        return ["healthy=0", "inner=1", "outer=2"]
    return ["IR=0", "OR=1", "BALL=2"]


def checkpoint_stem(label_mode: str):
    if label_mode == "inner_outer":
        return "cwru_inner_outer_lstm_fcn"
    if label_mode == "health_inner_outer":
        return "cwru_health_inner_outer_lstm_fcn"
    return "cwru_fault3_lstm_fcn"


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

    # overall accuracy
    acc = (p == y).mean()

    # balanced accuracy = mean recall
    recalls = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recall_i = tp / (tp + fn + 1e-9)
        recalls.append(recall_i)

    bal = float(np.mean(recalls))

    return acc, bal, cm


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = Path(__file__).resolve().parent.parent.parent
    data_root = Path(args.data_root)
    if args.label_mode == "health_inner_outer" and data_root.name == "12k_DE":
        data_root = data_root.parent
        print("[INFO] label_mode=health_inner_outer requires Normal files; using data root:", data_root)

    label_map = build_label_map(args.label_mode)

    train_loader, val_loader, test_loader, meta = build_cwru_from_folder(
        mat_dir=data_root,
        label_map=label_map,
        window_size=args.window_size,
        stride=args.stride,
        split=tuple(args.split),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_train=args.balance_train,
        normalize=args.normalize,
    )
    print("meta:", meta)

    xb, yb = next(iter(train_loader))
    print("batch x:", xb.shape, "batch y unique:", sorted(set(yb.tolist())))

    num_classes = meta["num_classes"]
    model = LSTMFCNClassifier(
        num_classes=num_classes,
        in_channels=1,
        lstm_hidden=128,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    os.makedirs(root / "checkpoints", exist_ok=True)
    os.makedirs(root / "logs", exist_ok=True)

    train_losses, val_accs, val_bals = [], [], []

    best_val_bal = -1.0
    stem = checkpoint_stem(args.label_mode)
    best_path = root / "checkpoints" / f"best_{stem}.pt"
    log_path = root / "logs" / f"{stem}_train_log.csv"
    print("label_mode:", args.label_mode)
    print("best checkpoint path:", best_path)
    print("train log path:", log_path)

    epochs = args.epochs
    for ep in range(1, epochs + 1):
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

        val_acc, val_bal, _ = eval_loader(model, val_loader, device, num_classes=num_classes)

        print(f"Epoch {ep:02d}/{epochs} train_loss={train_loss:.4f} | val_acc={val_acc:.4f} val_bal={val_bal:.4f}")

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_bals.append(val_bal)

        if val_bal > best_val_bal:
            best_val_bal = val_bal
            torch.save(model.state_dict(), best_path)
            print("  ✓ saved best")

    # save log
    log_df = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss": train_losses,
        "val_acc": val_accs,
        "val_bal": val_bals,
    })
    log_df.to_csv(log_path, index=False)
    print("Saved log to:", log_path)

    # test best
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, test_bal, test_cm = eval_loader(model, test_loader, device, num_classes=num_classes)

    print(f"[TEST] acc={test_acc:.4f} bal_acc={test_bal:.4f}")
    print(f"test confusion (rows=true, cols=pred), labels [{', '.join(label_mode_names(args.label_mode))}]:\n", test_cm)


if __name__ == "__main__":
    train()
