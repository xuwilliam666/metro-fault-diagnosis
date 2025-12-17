import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from pathlib import Path


def _load_1d_csv(path: Path) -> np.ndarray:
    """
    Load a CSV and flatten all numeric cells into 1D float32 array.
    Cleans '-', empty strings etc.
    """
    df = pd.read_csv(path)

    # common dirty tokens
    df = df.replace(["-", " -", "- ", "", " ", "  "], np.nan)

    # force numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    x = df.to_numpy(dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    return x


class MetroVibWindowDataset(Dataset):
    """
    Sliding window dataset for metro vibration data (3 axes).

    returns:
      x: FloatTensor [T, 3]
      y: LongTensor  scalar (0 normal, 1 failure)
    """
    def __init__(self, x_3axis: np.ndarray, y: int, window_size: int = 2048, stride: int = 512):
        assert x_3axis.ndim == 2 and x_3axis.shape[1] == 3, f"expected [N,3], got {x_3axis.shape}"
        self.x = x_3axis
        self.y = int(y)
        self.window_size = int(window_size)
        self.stride = int(stride)

        self.n = len(self.x)
        if self.n < self.window_size:
            raise ValueError(f"sequence too short: N={self.n} < window_size={self.window_size}")

        # valid start indices
        self.idxs = list(range(0, self.n - self.window_size + 1, self.stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.idxs[i]
        w = self.x[s:s + self.window_size]  # [T,3]
        return torch.from_numpy(w), torch.tensor(self.y, dtype=torch.long)


def build_metro_vib_dataloaders(
    x_fail, y_fail, z_fail,
    x_norm, y_norm, z_norm,
    window_size: int = 2048,
    stride: int = 512,
    split=(0.7, 0.15, 0.15),
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    balance_train: bool = True,
):
    """
    Key idea:
      1) read/clean each axis
      2) align length per class by min length across axes
      3) z-score using ALL data (fail+norm) to avoid leaking "time split" assumptions
      4) create full window datasets for fail and norm
      5) split by WINDOW INDICES randomly (not by time)
      6) optionally balance TRAIN with WeightedRandomSampler
    """
    # ---- 1) read & clean
    fail_x = _load_1d_csv(Path(x_fail))
    fail_y = _load_1d_csv(Path(y_fail))
    fail_z = _load_1d_csv(Path(z_fail))

    norm_x = _load_1d_csv(Path(x_norm))
    norm_y = _load_1d_csv(Path(y_norm))
    norm_z = _load_1d_csv(Path(z_norm))

    # ---- 2) align lengths per class (3 axes need same length)
    L_fail = min(len(fail_x), len(fail_y), len(fail_z))
    L_norm = min(len(norm_x), len(norm_y), len(norm_z))

    fail = np.stack([fail_x[:L_fail], fail_y[:L_fail], fail_z[:L_fail]], axis=1)  # [N,3]
    norm = np.stack([norm_x[:L_norm], norm_y[:L_norm], norm_z[:L_norm]], axis=1)

    # ---- 3) z-score using ALL samples (fail+norm)
    all_for_stats = np.concatenate([fail, norm], axis=0)
    mean = all_for_stats.mean(axis=0, keepdims=True)
    std = all_for_stats.std(axis=0, keepdims=True) + 1e-8

    fail = (fail - mean) / std
    norm = (norm - mean) / std

    # ---- 4) full window datasets
    ds_fail_all = MetroVibWindowDataset(fail, y=1, window_size=window_size, stride=stride)
    ds_norm_all = MetroVibWindowDataset(norm, y=0, window_size=window_size, stride=stride)

    # ---- 5) split by window indices
    rng = np.random.default_rng(seed)

    def split_indices(n, split_tuple):
        a, b, c = split_tuple
        if abs((a + b + c) - 1.0) > 1e-6:
            raise ValueError(f"split must sum to 1.0, got {split_tuple}")
        idx = np.arange(n)
        rng.shuffle(idx)
        n_tr = int(n * a)
        n_va = int(n * b)
        tr = idx[:n_tr]
        va = idx[n_tr:n_tr + n_va]
        te = idx[n_tr + n_va:]
        return tr, va, te

    fail_tr_idx, fail_va_idx, fail_te_idx = split_indices(len(ds_fail_all), split)
    norm_tr_idx, norm_va_idx, norm_te_idx = split_indices(len(ds_norm_all), split)

    ds_fail_tr = Subset(ds_fail_all, fail_tr_idx)
    ds_fail_va = Subset(ds_fail_all, fail_va_idx)
    ds_fail_te = Subset(ds_fail_all, fail_te_idx)

    ds_norm_tr = Subset(ds_norm_all, norm_tr_idx)
    ds_norm_va = Subset(ds_norm_all, norm_va_idx)
    ds_norm_te = Subset(ds_norm_all, norm_te_idx)

    train_ds = ConcatDataset([ds_fail_tr, ds_norm_tr])
    val_ds = ConcatDataset([ds_fail_va, ds_norm_va])
    test_ds = ConcatDataset([ds_fail_te, ds_norm_te])

    # ---- 6) train loader (optionally balanced)
    if balance_train:
        n_fail = len(ds_fail_tr)
        n_norm = len(ds_norm_tr)

        w_fail = 1.0 / max(n_fail, 1)
        w_norm = 1.0 / max(n_norm, 1)

        weights = torch.tensor([w_fail] * n_fail + [w_norm] * n_norm, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "L_fail": int(L_fail),
        "L_norm": int(L_norm),
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "num_test_windows": len(test_ds),
        "seed": int(seed),
        "window_size": int(window_size),
        "stride": int(stride),
        "split": tuple(split),
        "balance_train": bool(balance_train),
    }
    return train_loader, val_loader, test_loader, meta


# -----------------------
# Quick test + baselines
# -----------------------
def check_label_dist(loader, name, max_batches=None):
    c0 = c1 = 0
    for i, (_, y) in enumerate(loader):
        c0 += (y == 0).sum().item()
        c1 += (y == 1).sum().item()
        if max_batches is not None and (i + 1) >= max_batches:
            break
    tot = c0 + c1
    print(f"[{name}] total={tot}  fail(1)={c1} ({c1/tot:.3f})  norm(0)={c0} ({c0/tot:.3f})")


def simple_stat_features(loader, max_batches=200):
    feats, labels = [], []
    for i, (x, y) in enumerate(loader):
        x = x.numpy()  # [B,T,C]
        y = y.numpy()
        m = x.mean(axis=1)  # [B,C]
        s = x.std(axis=1)   # [B,C]
        f = np.concatenate([m, s], axis=1)  # [B,2C]
        feats.append(f)
        labels.append(y)
        if (i + 1) >= max_batches:
            break
    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(labels, axis=0)
    return X, Y


def eval_majority(loader, majority_label=1):
    y_all = []
    for _, y in loader:
        y_all.append(y.numpy())
    y_all = np.concatenate(y_all)
    y_pred = np.full_like(y_all, majority_label)

    acc = (y_pred == y_all).mean()

    tn = np.sum((y_all == 0) & (y_pred == 0))
    fp = np.sum((y_all == 0) & (y_pred == 1))
    fn = np.sum((y_all == 1) & (y_pred == 0))
    tp = np.sum((y_all == 1) & (y_pred == 1))

    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bal_acc = 0.5 * (tpr + tnr)

    print("Majority baseline acc:", float(acc))
    print("Majority baseline bal_acc:", float(bal_acc))
    print("confusion [[tn,fp],[fn,tp]] =", [[int(tn), int(fp)], [int(fn), int(tp)]])


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "MetroDataset"

    train_loader, val_loader, test_loader, meta = build_metro_vib_dataloaders(
        x_fail=root / "Failure" / "Metro_vibration_v1_x_axis_failure.csv",
        y_fail=root / "Failure" / "Metro_vibration_v1_y_axis_failure.csv",
        z_fail=root / "Failure" / "Metro_vibration_v1_z_axis_failure.csv",
        x_norm=root / "Normal" / "Metro_vibration_v1_x_axis_normal.csv",
        y_norm=root / "Normal" / "Metro_vibration_v1_y_axis_normal.csv",
        z_norm=root / "Normal" / "Metro_vibration_v1_z_axis_normal.csv",
        window_size=2048,
        stride=512,
        split=(0.7, 0.15, 0.15),
        batch_size=64,
        num_workers=0,
        seed=42,
        balance_train=True,
    )

    print("meta:", meta)

    x, y = next(iter(train_loader))
    print("batch x shape:", x.shape)  # [B,T,3]
    print("batch y:", y[:10].tolist())
    print("y unique:", sorted(set(y.tolist())))

    check_label_dist(train_loader, "train", max_batches=200)
    check_label_dist(val_loader, "val", max_batches=200)
    check_label_dist(test_loader, "test", max_batches=200)

    # baseline: mean/std features + logistic regression
    Xtr, Ytr = simple_stat_features(train_loader, max_batches=200)
    Xte, Yte = simple_stat_features(test_loader, max_batches=200)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, Ytr)

    y_pred = clf.predict(Xte)

    print("StatLogReg train acc:", float(clf.score(Xtr, Ytr)))
    print("StatLogReg test  acc:", float((y_pred == Yte).mean()))
    print("StatLogReg test bal_acc:", float(balanced_accuracy_score(Yte, y_pred)))
    print("confusion:\n", confusion_matrix(Yte, y_pred))

    # majority baseline
    eval_majority(test_loader, majority_label=1)