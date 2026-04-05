import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path

from src.data.metro_dataset import build_metro_vib_dataloaders
from src.models.lstm_fcn import LSTMFCNClassifier


def parse_args():
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Train Metro supervised baseline with LSTM-FCN")
    parser.add_argument("--data_root", type=Path, default=root / "data" / "raw" / "MetroDataset")
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--balance_train", action="store_true", default=True)
    parser.add_argument("--no_balance_train", action="store_false", dest="balance_train")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = Path(__file__).resolve().parent.parent.parent
    data_root = Path(args.data_root)

    train_loader, val_loader, test_loader, meta = build_metro_vib_dataloaders(
        x_fail=data_root / "Failure" / "Metro_vibration_v1_x_axis_failure.csv",
        y_fail=data_root / "Failure" / "Metro_vibration_v1_y_axis_failure.csv",
        z_fail=data_root / "Failure" / "Metro_vibration_v1_z_axis_failure.csv",
        x_norm=data_root / "Normal" / "Metro_vibration_v1_x_axis_normal.csv",
        y_norm=data_root / "Normal" / "Metro_vibration_v1_y_axis_normal.csv",
        z_norm=data_root / "Normal" / "Metro_vibration_v1_z_axis_normal.csv",
        window_size=args.window_size,
        stride=args.stride,
        split=tuple(args.split),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balance_train=args.balance_train,
    )
    print("meta:", meta)

    # quick batch check
    xb, yb = next(iter(train_loader))
    print("batch x:", xb.shape, "batch y:", yb[:10].tolist(), "y unique:", sorted(set(yb.tolist())))

    # Model
    model = LSTMFCNClassifier(
        num_classes=2,
        in_channels=3,
        lstm_hidden=128,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_bal_acc = -1.0
    os.makedirs(root / "checkpoints", exist_ok=True)

    def eval_loader(loader):
        model.eval()
        total = 0
        correct = 0

        # confusion
        tn = fp = fn = tp = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)           # [B,T,3]
                y = y.to(device)           # [B]

                logits = model(x)

                pred = logits.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

                tn += ((y == 0) & (pred == 0)).sum().item()
                fp += ((y == 0) & (pred == 1)).sum().item()
                fn += ((y == 1) & (pred == 0)).sum().item()
                tp += ((y == 1) & (pred == 1)).sum().item()

        acc = correct / total
        tpr = tp / (tp + fn + 1e-9)
        tnr = tn / (tn + fp + 1e-9)
        bal_acc = 0.5 * (tpr + tnr)

        return acc, bal_acc, (tn, fp, fn, tp)

    num_epochs = args.epochs
    for epoch in range(1, num_epochs + 1):
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

        train_loss = total_loss / total
        val_acc, val_bal, val_conf = eval_loader(val_loader)

        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} val_bal_acc={val_bal:.4f} "
            f"conf(tn,fp,fn,tp)={val_conf}"
        )

        if val_bal > best_bal_acc:
            best_bal_acc = val_bal
            torch.save(model.state_dict(), root / "checkpoints" / "best_metro_lstm_fcn.pt")
            print("  ✓ saved best")

    # Test best
    model.load_state_dict(torch.load(root / "checkpoints" / "best_metro_lstm_fcn.pt", map_location=device))
    test_acc, test_bal, test_conf = eval_loader(test_loader)
    print(f"[TEST] acc={test_acc:.4f} bal_acc={test_bal:.4f} conf(tn,fp,fn,tp)={test_conf}")


if __name__ == "__main__":
    train()
