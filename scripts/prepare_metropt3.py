import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).parent.parent
    csv_path = root / "data" / "raw" / "MetroPT" / "MetroPT3(AirCompressor).csv"
    out_dir = root / "data" / "processed" / "MetroPT"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1) timestamp
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if not ts_candidates:
        raise RuntimeError("No timestamp-like column found (need a time/date column).")
    ts_col = ts_candidates[0]

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    intervals = [
        ("04/18/2020 00:00", "04/18/2020 23:59"),
        ("05/29/2020 23:30", "05/30/2020 06:00"),
        ("06/05/2020 10:00", "06/07/2020 14:30"),
        ("07/15/2020 14:30", "07/15/2020 19:00"),
    ]
    intervals = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in intervals]

    label = np.zeros(len(df), dtype=np.int8)
    for (s, e) in intervals:
        mask = (df[ts_col] >= s) & (df[ts_col] <= e)
        label[mask.values] = 1

    first_fail_start = intervals[0][0]
    split_idx = int((df[ts_col] < first_fail_start).sum())

    feat_cols = [c for c in df.columns if c != ts_col]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)

    X_train_raw = X[:split_idx]
    col_mean = np.nanmean(X_train_raw, axis=0, keepdims=True)
    X = np.where(np.isfinite(X), X, col_mean)

    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    train = Xn[:split_idx].astype(np.float32)
    test = Xn[split_idx:].astype(np.float32)
    test_label = label[split_idx:].astype(np.int8)

    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "test.npy", test)
    np.save(out_dir / "test_label.npy", test_label)

    meta = {
        "timestamp_col": ts_col,
        "feature_cols": feat_cols,
        "num_features": int(train.shape[1]),
        "split_idx": int(split_idx),
        "train_len": int(train.shape[0]),
        "test_len": int(test.shape[0]),
        "test_anomaly_ratio": float(test_label.mean()),
        "failure_intervals": [(str(s), str(e)) for (s, e) in intervals],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", out_dir)
    print("train:", train.shape, "test:", test.shape, "test_label:", test_label.shape)
    print("test anomaly ratio:", test_label.mean())


if __name__ == "__main__":
    main()