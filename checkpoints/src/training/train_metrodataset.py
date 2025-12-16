import os
import torch
import torch.nn as nn
from pathlib import Path

from src.data.metro_dataset import build_metro_vib_dataloaders
from src.models.lstm_fcn import LSTMFCNClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = Path(__file__).resolve().parent.parent.parent
    data_root = root / "data" / "raw" / "MetroDataset"

    train_loader, val_loader, test_loader, meta = build_metro_vib_dataloaders(
        x_fail=data_root / "Failure" / "Metro_vibration_v1_x_axis_failure.csv",
        y_fail=data_root / "Failure" / "Metro_vibration_v1_y_axis_failure.csv",
        z_fail=data_root / "Failure" / "Metro_vibration_v1_z_axis_failure.csv",
        x_norm=data_root / "Normal" / "Metro_vibration_v1_x_axis_normal.csv",
        y_norm=data_root / "Normal" / "Metro_vibration_v1_y_axis_normal.csv",
        z_norm=data_root / "Normal" / "Metro_vibration_v1_z_axis_normal.csv",
        window_size=2048,
        stride=512,
        split=(0.7, 0.15, 0.15),
        batch_size=64,
        num_workers=0,
        seed=42,
        balance_train=True,
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

                try:
                    logits = model(x)      # preferred
                except Exception:
                    logits = model(x.permute(0, 2, 1))  # [B,3,T]

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

    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            try:
                logits = model(x)
            except Exception:
                logits = model(x.permute(0, 2, 1))

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