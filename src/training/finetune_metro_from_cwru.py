import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data.metro_dataset import build_metro_vib_dataloaders
from src.models.lstm_fcn import LSTMFCNClassifier
from src.training.transfer_utils import freeze_feature_extractor, load_cwru_pretrained_weights


def parse_args():
    root = Path(__file__).resolve().parent.parent.parent
    default_data_root = root / "data" / "raw" / "MetroDataset"
    default_output_dir = root / "logs" / "metro_transfer"

    parser = argparse.ArgumentParser(description="Finetune Metro LSTM-FCN from a CWRU checkpoint")
    parser.add_argument("--cwru_ckpt", type=Path, required=True, help="Path to pretrained CWRU checkpoint")
    parser.add_argument("--data_root", type=Path, default=default_data_root, help="MetroDataset root directory")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir, help="Directory for metrics and checkpoints")
    parser.add_argument("--mode", type=str, choices=["full", "frozen"], default="full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance_train", action="store_true", default=True)
    parser.add_argument("--no_balance_train", action="store_false", dest="balance_train")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    data_root = Path(args.data_root)
    return build_metro_vib_dataloaders(
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


def evaluate(model, loader, device, num_classes=2):
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

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def format_metrics(name, metrics):
    return (
        f"[{name}] "
        f"acc={metrics['accuracy']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"precision_macro={metrics['precision_macro']:.4f} "
        f"recall_macro={metrics['recall_macro']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} "
        f"confusion={metrics['confusion_matrix']}"
    )


def args_to_dict(args):
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, meta = build_loaders(args)
    print("meta:", meta)

    xb, yb = next(iter(train_loader))
    print("batch x:", xb.shape, "batch y:", yb[:10].tolist(), "y unique:", sorted(set(yb.tolist())))

    model = LSTMFCNClassifier(
        num_classes=2,
        in_channels=3,
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
    ).to(device)

    model = load_cwru_pretrained_weights(model, args.cwru_ckpt, map_location=device)
    if args.mode == "frozen":
        freeze_feature_extractor(model, freeze=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = Path(args.output_dir) / f"best_metro_from_cwru_{args.mode}.pt"
    metrics_path = Path(args.output_dir) / f"metrics_{args.mode}.json"
    metrics_txt_path = Path(args.output_dir) / f"metrics_{args.mode}.txt"
    cm_path = Path(args.output_dir) / f"confusion_{args.mode}.npy"

    best_val_bal = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
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
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} | "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_confusion={val_metrics['confusion_matrix']}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
        )

        if val_metrics["balanced_accuracy"] > best_val_bal:
            best_val_bal = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": args_to_dict(args),
                    "meta": meta,
                    "best_val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
            print("  ✓ saved best")

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print(format_metrics("TRAIN", train_metrics))
    print(format_metrics("VAL", val_metrics))
    print(format_metrics("TEST", test_metrics))

    report = {
        "mode": args.mode,
        "cwru_ckpt": str(args.cwru_ckpt),
        "data_root": str(args.data_root),
        "best_checkpoint": str(best_ckpt_path),
        "args": args_to_dict(args),
        "meta": meta,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(metrics_txt_path, "w") as f:
        f.write(format_metrics("TRAIN", train_metrics) + "\n")
        f.write(format_metrics("VAL", val_metrics) + "\n")
        f.write(format_metrics("TEST", test_metrics) + "\n")

    np.save(cm_path, np.asarray(test_metrics["confusion_matrix"], dtype=np.int64))

    print("Saved artifacts:")
    print(" -", best_ckpt_path)
    print(" -", metrics_path)
    print(" -", metrics_txt_path)
    print(" -", cm_path)


if __name__ == "__main__":
    main()
