import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.metro_dataset import build_metro_vib_dataloaders
from src.models.lstm_fcn import LSTMFCNClassifier


def parse_args():
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Extract Metro LSTM-FCN features for analysis")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to trained Metro model checkpoint")
    parser.add_argument("--output", type=Path, default=root / "logs" / "metro_features.npz")
    parser.add_argument("--data_root", type=Path, default=root / "data" / "raw" / "MetroDataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def collect_features(model, loader: DataLoader, device):
    feats, labels = [], []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = model.forward_features(x)
            feats.append(feat.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, meta = build_metro_vib_dataloaders(
        x_fail=Path(args.data_root) / "Failure" / "Metro_vibration_v1_x_axis_failure.csv",
        y_fail=Path(args.data_root) / "Failure" / "Metro_vibration_v1_y_axis_failure.csv",
        z_fail=Path(args.data_root) / "Failure" / "Metro_vibration_v1_z_axis_failure.csv",
        x_norm=Path(args.data_root) / "Normal" / "Metro_vibration_v1_x_axis_normal.csv",
        y_norm=Path(args.data_root) / "Normal" / "Metro_vibration_v1_y_axis_normal.csv",
        z_norm=Path(args.data_root) / "Normal" / "Metro_vibration_v1_z_axis_normal.csv",
        window_size=args.window_size,
        stride=args.stride,
        split=tuple(args.split),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balance_train=False,
    )
    print("meta:", meta)

    model = LSTMFCNClassifier(num_classes=2, in_channels=3).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))

    train_feat, train_y = collect_features(model, train_loader, device)
    val_feat, val_y = collect_features(model, val_loader, device)
    test_feat, test_y = collect_features(model, test_loader, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        train_features=train_feat,
        train_labels=train_y,
        val_features=val_feat,
        val_labels=val_y,
        test_features=test_feat,
        test_labels=test_y,
    )
    print("Saved features to:", args.output)


if __name__ == "__main__":
    main()
