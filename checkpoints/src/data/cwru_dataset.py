import os
from pathlib import Path
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def load_cwru_mat_1d(mat_path: Path) -> np.ndarray:
    """
    Load a CWRU .mat file and return the longest 1D numeric array as signal.
    Avoid hardcoding variable names like X097_DE_time, etc.
    """
    mdict = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    candidates = []
    for k, v in mdict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            arr = v
            if arr.ndim == 1:
                candidates.append(arr.astype(np.float32))
            elif arr.ndim == 2 and (1 in arr.shape):
                candidates.append(arr.reshape(-1).astype(np.float32))

    if not candidates:
        raise ValueError(f"No numeric 1D arrays found in: {mat_path}")

    sig = max(candidates, key=lambda a: a.size).reshape(-1)
    sig = sig[np.isfinite(sig)]
    if sig.size < 10:
        raise ValueError(f"Signal too short after cleaning: {mat_path}")
    return sig


def zscore_1d(x: np.ndarray, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = float(np.mean(x))
    if std is None:
        std = float(np.std(x))
    x = (x - mean) / (std + eps)
    return x.astype(np.float32), mean, std


class CWRUWindowDataset(Dataset):
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
                w = sig[s:s + self.window_size]  # [T]
                X_list.append(w)
                y_list.append(int(lab))

        self.X = np.stack(X_list, axis=0) if len(X_list) else np.zeros((0, self.window_size), np.float32)
        self.y = np.asarray(y_list, dtype=np.int64)

    def __len__(self):
        return int(self.y.size)

    def __getitem__(self, idx):
        x = self.X[idx]                          # [T]
        x = torch.from_numpy(x).unsqueeze(-1)    # [T,1]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def build_cwru_from_folder(
    mat_dir: Path,
    label_map: dict,
    window_size=1024,
    stride=256,
    split=(0.7, 0.15, 0.15),
    seed=42,
    batch_size=64,
    num_workers=0,
    balance_train=True,
    normalize="per_file",  # "per_file" or "train_global"
):
    """
    mat_dir: folder containing .mat files (we search recursively)
    label_map: {keyword_in_filename: class_id}
      - matched by "keyword in filename" (case-insensitive)
      - for A(3-class): {"ir":0, "or":1, "b007":2, ...}

    Split is FILE-level (important! avoid window leakage).
    """
    rng = np.random.default_rng(seed)
    mat_dir = Path(mat_dir)

    files = sorted([p for p in mat_dir.rglob("*.mat")])
    if not files:
        raise FileNotFoundError(f"No .mat files found under: {mat_dir}")

    signals, labels, used_files = [], [], []

    for fp in files:
        name = fp.name.lower()

        lab = None
        for key, val in label_map.items():
            if key.lower() in name:
                lab = int(val)
                break
        if lab is None:
            continue

        sig = load_cwru_mat_1d(fp)

        if normalize == "per_file":
            sig, _, _ = zscore_1d(sig)

        signals.append(sig)
        labels.append(lab)
        used_files.append(fp.name)

    if not signals:
        raise ValueError("No files matched your label_map. Check naming rules.")

    # file-level STRATIFIED split
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
        va_idx.extend(cls_idx[n_tr : n_tr + n_va].tolist())
        te_idx.extend(cls_idx[n_tr + n_va :].tolist())


    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    rng.shuffle(te_idx)

    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    va_idx = np.asarray(va_idx, dtype=np.int64)
    te_idx = np.asarray(te_idx, dtype=np.int64)


    sig_tr = [signals[i] for i in tr_idx]
    lab_tr = [labels[i]  for i in tr_idx]
    sig_va = [signals[i] for i in va_idx]
    lab_va = [labels[i]  for i in va_idx]
    sig_te = [signals[i] for i in te_idx]
    lab_te = [labels[i]  for i in te_idx]

    mean_std = None
    if normalize == "train_global":
        all_tr = np.concatenate(sig_tr, axis=0)
        all_tr, m, s = zscore_1d(all_tr)
        sig_tr = [(x - m) / (s + 1e-8) for x in sig_tr]
        sig_va = [(x - m) / (s + 1e-8) for x in sig_va]
        sig_te = [(x - m) / (s + 1e-8) for x in sig_te]
        mean_std = (float(m), float(s))

    # window datasets
    train_ds = CWRUWindowDataset(sig_tr, lab_tr, window_size=window_size, stride=stride)
    val_ds   = CWRUWindowDataset(sig_va, lab_va, window_size=window_size, stride=stride)
    test_ds  = CWRUWindowDataset(sig_te, lab_te, window_size=window_size, stride=stride)

    # compute num_classes from label_map (A: should be 3)
    all_classes = sorted(set(label_map.values()))
    num_classes = len(all_classes)

    # dataloaders (balanced sampling on TRAIN)
    if balance_train and len(train_ds) > 0:
        y = train_ds.y
        classes, counts = np.unique(y, return_counts=True)
        w_per_class = {int(c): 1.0 / float(cnt) for c, cnt in zip(classes, counts)}
        weights = np.array([w_per_class[int(yy)] for yy in y], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)

    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "num_files_total": len(signals),
        "num_files_train": len(sig_tr),
        "num_files_val": len(sig_va),
        "num_files_test": len(sig_te),
        "num_classes": int(num_classes),
        "class_ids": all_classes,
        "window_size": int(window_size),
        "stride": int(stride),
        "split": split,
        "seed": seed,
        "normalize": normalize,
        "mean_std": mean_std,
        "balance_train": bool(balance_train),
        "used_files_sample": used_files[:10],
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta