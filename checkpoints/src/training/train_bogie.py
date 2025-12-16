import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.bogie_dataset import create_dataloaders
from src.models.lstm_fcn import LSTMFCNClassifier

DEBUG = True
root = Path(__file__).resolve().parent.parent.parent

def train_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data paths

    train_csv = root.joinpath("data", "processed", "bogie", "dataset1_train.csv")
    val_csv = root.joinpath("data", "processed", "bogie", "dataset1_val.csv")
    test_csv = root.joinpath("data", "processed", "bogie", "dataset1_test.csv")

    # Print number of classes
    df_train = pd.read_csv(train_csv)
    num_classes = df_train['label'].nunique()
    print("Number of classes: ", num_classes)

    # Data Loaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        batch_size=64,
        num_workers=0,
        use_fft=False,
    )

    # Batch shape check
    sample_batch, sample_labels = next(iter(train_loader))
    print("Sample batch shape:", sample_batch.shape)
    print("Sample labels:", sample_labels[:10].tolist())

    # Model, Loss, Optimizer
    model = LSTMFCNClassifier(
        num_classes=num_classes, lstm_hidden=128,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 50
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    # Training Loop
    for epoch in range(1, num_epochs + 1):

        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)  # [B, T, F]
            y = y.to(device)  # [B]

            logits = model(x)  # [B, num_classes]
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                logits_val = model(x_val)
                loss_val = criterion(logits_val, y_val)

                val_loss_total += loss_val.item() * x_val.size(0)
                preds_val = logits_val.argmax(dim=1)
                val_correct += (preds_val == y_val).sum().item()
                val_total += y_val.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), root / "checkpoints" / "best_lstm_fcn_fft_dataset1.pt")
            print("  ✓ Saved best model")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (patience {patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # Load best model before testing
    model.load_state_dict(
        torch.load(root / "checkpoints" / "best_lstm_fcn_fft_dataset1.pt", map_location=device)
    )
    model.to(device)
    # Test Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            logits_test = model(x_test)
            preds_test = logits_test.argmax(dim=1)
            test_correct += (preds_test == y_test).sum().item()
            test_total += y_test.size(0)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save Model
    os.makedirs(root / "checkpoints", exist_ok=True)
    torch.save(model.state_dict(), root / "checkpoints" / "lstm_fcn_dataset1.pt")
    print("Saved model to checkpoints/lstm_fcn_dataset1.pt")


if __name__ == "__main__":
    train_classifier()
