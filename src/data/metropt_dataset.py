from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    """
    For Anomaly Transformer:
      x: FloatTensor [win_size, num_features]
    """
    def __init__(self, arr_2d: np.ndarray, win_size: int = 100, step: int = 10):
        assert arr_2d.ndim == 2, f"need [T,D], got {arr_2d.shape}"
        self.x = arr_2d.astype(np.float32)
        self.win_size = int(win_size)
        self.step = int(step)

        T = self.x.shape[0]
        if T < self.win_size:
            self.idxs = []
        else:
            self.idxs = list(range(0, T - self.win_size + 1, self.step))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.idxs[i]
        w = self.x[s:s + self.win_size]  # [win, D]
        return torch.from_numpy(w)       # FloatTensor


def build_metropt3_loaders(
    processed_dir: str | Path,
    win_size: int = 100,
    step: int = 10,
    batch_size: int = 128,
    num_workers: int = 0,
):
    """
    Expect Step1 outputs:
      processed_dir/train.npy  [T_train, D]
      processed_dir/test.npy   [T_test, D]
      processed_dir/test_label.npy [T_test]
    """
    processed_dir = Path(processed_dir)
    train = np.load(processed_dir / "train.npy")       # [T,D]
    test  = np.load(processed_dir / "test.npy")        # [T,D]
    test_label = np.load(processed_dir / "test_label.npy")  # [T]

    train_ds = SlidingWindowDataset(train, win_size=win_size, step=step)
    test_ds  = SlidingWindowDataset(test,  win_size=win_size, step=step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )

    meta = {
        "train_T": int(train.shape[0]),
        "test_T": int(test.shape[0]),
        "D": int(train.shape[1]),
        "win_size": int(win_size),
        "step": int(step),
        "num_train_windows": len(train_ds),
        "num_test_windows": len(test_ds),
        "test_label_ratio": float(test_label.mean()),
    }
    return train_loader, test_loader, test_label, meta

if __name__ == "__main__":
    # Quick test
    root =  Path(__file__).parent.parent.parent
    train_loader, test_loader, test_label, meta = build_metropt3_loaders(
        processed_dir=root / "data" / "processed" /"MetroPT",
        win_size=100,
        step=10,
        batch_size=64,
        num_workers=0,
    )
    print("meta:", meta)

    xb = next(iter(train_loader))
    print("train batch:", xb.shape)  # [B, win, D]
    xb2 = next(iter(test_loader))
    print("test batch:", xb2.shape)  # [B, win, D]
    print("test_label shape:", test_label.shape, "ratio:", test_label.mean())