from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _collect_numeric_1d_arrays(obj, out):
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number):
        if obj.ndim == 1:
            out.append(obj.astype(np.float32).reshape(-1))
        elif obj.ndim == 2 and 1 in obj.shape:
            out.append(obj.astype(np.float32).reshape(-1))
        return

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            _collect_numeric_1d_arrays(item, out)
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("__"):
                continue
            _collect_numeric_1d_arrays(value, out)
        return

    if hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            _collect_numeric_1d_arrays(value, out)


def load_paderborn_mat_1d(mat_path: Path) -> np.ndarray:
    """
    Load a Paderborn .mat file and return the longest numeric 1D array.
    The files can contain nested matlab structs, so recurse conservatively.
    """
    mdict = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    candidates = []
    _collect_numeric_1d_arrays(mdict, candidates)

    if not candidates:
        raise ValueError(f"No numeric 1D arrays found in: {mat_path}")

    sig = max(candidates, key=lambda a: a.size).reshape(-1)
    sig = sig[np.isfinite(sig)]
    if sig.size < 10:
        raise ValueError(f"Signal too short after cleaning: {mat_path}")
    return sig.astype(np.float32)


def zscore_1d(x: np.ndarray, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = float(np.mean(x))
    if std is None:
        std = float(np.std(x))
    x = (x - mean) / (std + eps)
    return x.astype(np.float32), mean, std


class PaderbornWindowDataset(Dataset):
    """
    Each item:
      x: FloatTensor [T, 1]
      y: LongTensor  []
    """
    def __init__(self, signals, labels, window_size=1024, stride=256):
        assert len(signals) == len(labels)
        self.window_size = int(window_size)
        self.stride = int(stride)

        X_list = []
        y_list = []

        for sig, lab in zip(signals, labels):
            sig = np.asarray(sig, dtype=np.float32).reshape(-1)
            n = sig.size
            if n < self.window_size:
                continue
            for s in range(0, n - self.window_size + 1, self.stride):
                w = sig[s:s + self.window_size]
                X_list.append(w)
                y_list.append(int(lab))

        self.X = np.stack(X_list, axis=0) if len(X_list) else np.zeros((0, self.window_size), np.float32)
        self.y = np.asarray(y_list, dtype=np.int64)

    def __len__(self):
        return int(self.y.size)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(-1)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def infer_paderborn_label(path: Path):
    """
    First-pass 3-class mapping aligned to CWRU:
      K00* -> healthy
      KI*  -> inner race fault
      KA*  -> outer race fault
    Skip KB* for now because it does not map cleanly to the initial CWRU setup.
    """
    folder = path.parent.name.upper()
    if folder.startswith("K0"):
        return 0
    if folder.startswith("KI"):
        return 1
    if folder.startswith("KA"):
        return 2
    return None


def build_paderborn_from_folder(
    data_dir: Path,
    window_size=2048,
    stride=512,
    split=(0.7, 0.15, 0.15),
    seed=42,
    batch_size=64,
    num_workers=0,
    balance_train=True,
    normalize="per_file",  # "per_file" or "train_global"
    include_conditions=None,
):
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)

    files = sorted([p for p in data_dir.rglob("*.mat") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .mat files found under: {data_dir}")

    signals, labels, used_files = [], [], []
    skipped_files = []

    for fp in files:
        label = infer_paderborn_label(fp)
        if label is None:
            continue

        if include_conditions is not None:
            prefix = "_".join(fp.stem.split("_")[:3])
            if prefix not in include_conditions:
                continue

        try:
            sig = load_paderborn_mat_1d(fp)
        except (OSError, ValueError, TypeError, NotImplementedError) as exc:
            skipped_files.append({"file": str(fp.relative_to(data_dir)), "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[WARN] skipping unreadable Paderborn file: {fp.name} ({type(exc).__name__}: {exc})")
            continue

        if normalize == "per_file":
            sig, _, _ = zscore_1d(sig)

        signals.append(sig)
        labels.append(label)
        used_files.append(str(fp.relative_to(data_dir)))

    if not signals:
        raise ValueError("No usable Paderborn files were found.")

    present_classes = sorted(set(labels))
    expected_classes = [0, 1, 2]
    if present_classes != expected_classes:
        raise ValueError(
            f"After filtering unreadable Paderborn files, class coverage is incomplete. "
            f"expected={expected_classes}, got={present_classes}. "
            f"skipped_files={len(skipped_files)}"
        )

    a, b, c = split
    labels_arr = np.asarray(labels, dtype=np.int64)
    tr_idx, va_idx, te_idx = [], [], []

    for cls in sorted(set(labels_arr.tolist())):
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)

        n_cls = len(cls_idx)
        n_tr = int(n_cls * a)
        n_va = int(n_cls * b)

        if n_cls >= 3:
            n_tr = max(n_tr, 1)
            n_va = max(n_va, 1)
            if n_tr + n_va >= n_cls:
                n_va = max(1, n_cls - n_tr - 1)

        tr_idx.extend(cls_idx[:n_tr].tolist())
        va_idx.extend(cls_idx[n_tr:n_tr + n_va].tolist())
        te_idx.extend(cls_idx[n_tr + n_va:].tolist())

    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    rng.shuffle(te_idx)

    sig_tr = [signals[i] for i in tr_idx]
    lab_tr = [labels[i] for i in tr_idx]
    sig_va = [signals[i] for i in va_idx]
    lab_va = [labels[i] for i in va_idx]
    sig_te = [signals[i] for i in te_idx]
    lab_te = [labels[i] for i in te_idx]

    mean_std = None
    if normalize == "train_global":
        all_tr = np.concatenate(sig_tr, axis=0)
        all_tr, m, s = zscore_1d(all_tr)
        sig_tr = [(x - m) / (s + 1e-8) for x in sig_tr]
        sig_va = [(x - m) / (s + 1e-8) for x in sig_va]
        sig_te = [(x - m) / (s + 1e-8) for x in sig_te]
        mean_std = (float(m), float(s))

    train_ds = PaderbornWindowDataset(sig_tr, lab_tr, window_size=window_size, stride=stride)
    val_ds = PaderbornWindowDataset(sig_va, lab_va, window_size=window_size, stride=stride)
    test_ds = PaderbornWindowDataset(sig_te, lab_te, window_size=window_size, stride=stride)

    if balance_train and len(train_ds) > 0:
        y = train_ds.y
        classes, counts = np.unique(y, return_counts=True)
        w_per_class = {int(c): 1.0 / float(cnt) for c, cnt in zip(classes, counts)}
        weights = np.array([w_per_class[int(yy)] for yy in y], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "num_files_total": len(signals),
        "num_files_train": len(sig_tr),
        "num_files_val": len(sig_va),
        "num_files_test": len(sig_te),
        "num_classes": 3,
        "class_ids": [0, 1, 2],
        "class_names": {0: "healthy", 1: "inner", 2: "outer"},
        "window_size": int(window_size),
        "stride": int(stride),
        "split": split,
        "seed": seed,
        "normalize": normalize,
        "mean_std": mean_std,
        "balance_train": bool(balance_train),
        "used_files_sample": used_files[:10],
        "skipped_files_count": len(skipped_files),
        "skipped_files_sample": skipped_files[:10],
        "include_conditions": sorted(include_conditions) if include_conditions is not None else None,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta
