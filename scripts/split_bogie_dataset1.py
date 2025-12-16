import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    bogie_dataset1 = root.joinpath("data", "raw", "bogie", "dataset1.csv")
    out_dir = root.joinpath("data", "processed", "bogie")
    os.makedirs(out_dir, exist_ok=True) if not out_dir.exists() else None
    df = pd.read_csv(bogie_dataset1)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df["label"]
    )
    print("Total samples:", len(df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    train_path = os.path.join(out_dir, "dataset1_train.csv")
    val_path = os.path.join(out_dir, "dataset1_val.csv")
    test_path = os.path.join(out_dir, "dataset1_test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved:")
    print("  ", train_path)
    print("  ", val_path)
    print("  ", test_path)


if __name__ == "__main__":
    main()
