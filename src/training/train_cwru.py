import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from src.data.cwru_dataset import build_cwru_from_folder
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = Path(__file__).resolve().parent.parent.parent
    data_root = root / "data" / "raw" / "CWRU" / "12k_DE"

    # A: 3-class
    # IR=0, OR=1, BALL=2
    label_map = {
        "ir": 0,
        "or": 1,
        "b007": 2,
        "b014": 2,
        "b021": 2,
        "b028": 2,
    }

    train_loader, val_loader, test_loader, meta = build_cwru_from_folder(
        mat_dir=data_root,
        label_map=label_map,
        window_size=500,
        stride=250,
        split=(0.7, 0.15, 0.15),
        seed=42,
        batch_size=64,
        num_workers=0,
        balance_train=True,
        normalize="per_file",
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
    best_path = root / "checkpoints" / "best_cwru_lstm_fcn.pt"
    log_path  = root / "logs" / "cwru_train_log.csv"

    epochs = 20
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
    print("test confusion (rows=true, cols=pred), labels [IR=0, OR=1, BALL=2]:\n", test_cm)


if __name__ == "__main__":
    train()