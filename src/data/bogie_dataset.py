import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class BogieDataset(Dataset):
    """
    In this dataset, each row corresponds to a bogie sample.
    Columns include conditions, vibration and label.
    """

    def __init__(self, csv_path, feature_cols=None, label_col="label", scaler=None):
        """
        Args:
            csv_path (Path): Path to the CSV file containing the dataset.
            feature_cols (list of str): List of column names to be used as features.
                                        If None, all columns except label_col are used.
            label_col (str): Column name for the label.
            scaler (sklearn.preprocessing scaler): Scaler to normalize features.
                                                   If None, no scaling is applied.
        """
        self.df = pd.read_csv(csv_path)

        if feature_cols is None:
            feature_cols = [
                col for col in self.df.columns if col.startswith("vibration_")
            ]

        self.feature_cols = feature_cols
        self.label_col = label_col
        x = self.df[feature_cols].values.astype(np.float32)
        y = self.df[label_col].values.astype(np.int64)

        if scaler is None:
            self.scaler = StandardScaler().fit(x)
        else:
            self.scaler = scaler

        x = self.scaler.transform(x)
        x = x.reshape(
            len(x), len(feature_cols), 1
        )  # Reshape into [batch size, num features, label]
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_dataloaders(train_csv, val_csv, test_csv, batch_size=64, num_workers=4):
    df_train = pd.read_csv(train_csv)
    feature_cols = [c for c in df_train.columns if c.startswith("vibration_")]
    scaler = StandardScaler().fit(df_train[feature_cols].values.astype(np.float32))

    train_ds = BogieDataset(train_csv, feature_cols, "label", scaler=scaler)
    val_ds = BogieDataset(val_csv, feature_cols, "label", scaler=scaler)
    test_ds = BogieDataset(test_csv, feature_cols, "label", scaler=scaler)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, scaler
