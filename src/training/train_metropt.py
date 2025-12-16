import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
from src.data.metropt_dataset import build_metropt3_loaders
from src.models.anomaly_transformer import AnomalyTransformer, anomaly_transformer_loss


@torch.no_grad()
def compute_scores(model, loader, device, temperature=50.0):
    """
    Return WINDOW-level scores [num_windows]
    score = sum_t softmax(-disc_t / T) * rec_t
    where disc_t is association discrepancy per time point.
    """
    model.eval()
    all_scores = []

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)  # [B,T,D]

        recon, series_list, prior_list = model(x)

        # rec per time point: [B,T]
        rec_t = (recon - x).pow(2).mean(dim=-1)

        # disc per time point: [B,T]
        disc_t = 0.0
        eps = 1e-8
        for s, p in zip(series_list, prior_list):
            s_ = s.clamp_min(eps)  # [B,H,T,T]
            p_ = p.clamp_min(eps)

            # KL along last dim (keys) => [B,H,T]
            kl_sp_t = (s_ * (torch.log(s_) - torch.log(p_))).sum(dim=-1)
            kl_ps_t = (p_ * (torch.log(p_) - torch.log(s_))).sum(dim=-1)

            # average heads => [B,T]
            disc_t = disc_t + (kl_sp_t + kl_ps_t).mean(dim=1)

        # weight: [B,T]
        w = F.softmax(-disc_t / temperature, dim=-1)

        # window score: [B]
        score = (w * rec_t).sum(dim=-1)
        all_scores.append(score.detach().cpu().numpy())

    return np.concatenate(all_scores, axis=0)


def window_to_point_scores(win_scores, T, win_size, step):
    """
    Spread window scores back to point-level by averaging overlapping windows.
    Output length == T
    """
    ps = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)

    i = 0
    for s in range(0, T - win_size + 1, step):
        ps[s:s + win_size] += win_scores[i]
        cnt[s:s + win_size] += 1.0
        i += 1

    ps /= np.maximum(cnt, 1.0)
    return ps


def point_adjust(pred, label):
    """
    Xu et al. 2018 "point-adjust":
    If any point in a GT anomaly segment is predicted as anomaly,
    mark the whole GT segment as predicted anomaly.
    """
    pred = pred.astype(np.int32).copy()
    label = label.astype(np.int32)

    in_anom = False
    start = 0

    for i in range(len(label)):
        if label[i] == 1 and not in_anom:
            in_anom = True
            start = i

        if in_anom and (i == len(label) - 1 or label[i + 1] == 0):
            end = i
            if pred[start:end + 1].max() == 1:
                pred[start:end + 1] = 1
            in_anom = False

    return pred


def best_f1_with_adjust(point_scores, label, n_thr=300, q_lo=0.80, q_hi=0.999):
    """
    Sweep thresholds and return best (P,R,F1,thr) using point-adjust.
    """
    label = label.astype(np.int32)

    thrs = np.quantile(point_scores, np.linspace(q_lo, q_hi, n_thr))

    best = {"thr": None, "P": 0.0, "R": 0.0, "F1": -1.0}
    best_pred = None

    for thr in thrs:
        pred = (point_scores > thr).astype(np.int32)
        pred_adj = point_adjust(pred, label)

        P, R, F1, _ = precision_recall_fscore_support(
            label, pred_adj, average="binary", zero_division=0
        )

        if F1 > best["F1"]:
            best = {"thr": float(thr), "P": float(P), "R": float(R), "F1": float(F1)}
            best_pred = pred_adj

    return best, best_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = Path(__file__).resolve().parent.parent.parent

    train_loader, test_loader, test_label, meta = build_metropt3_loaders(
        processed_dir=root / "data" / "processed" / "MetroPT",
        win_size=100,
        step=10,
        batch_size=64,
        num_workers=0
    )
    print("meta:", meta)

    xb = next(iter(train_loader))
    if isinstance(xb, (list, tuple)):
        xb = xb[0]
    print("train batch:", xb.shape)

    D = meta["D"]
    model = AnomalyTransformer(
        c_in=D,
        d_model=128,
        n_heads=4,
        e_layers=3,
        d_ff=256,
        dropout=0.1
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    os.makedirs(root / "checkpoints", exist_ok=True)
    os.makedirs(root / "logs", exist_ok=True)

    best_loss = 1e18
    epochs = 5
    lambda_kl = 1.0

    # TRAIN
    history = {"epoch": [], "loss": [], "rec": [], "seriesKL": [], "priorKL": []}
    for ep in range(1, epochs + 1):
        model.train()
        losses, recs, sds, pds = [], [], [], []

        for batch in train_loader:
            x = (
                batch.to(device)
                if not isinstance(batch, (list, tuple))
                else batch[0].to(device)
            )

            # ====== Phase A: MIN (update prior, series detached in loss) ======
            recon, series_list, prior_list = model(x)
            loss_min, loss_max, rec_loss, series_loss, prior_loss = (
                anomaly_transformer_loss(
                    x, recon, series_list, prior_list, lambda_kl=lambda_kl
                )
            )

            optim.zero_grad()
            loss_min.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # ====== Phase B: MAX (update series, prior detached in loss) ======
            recon, series_list, prior_list = model(x)
            loss_min2, loss_max2, rec_loss2, series_loss2, prior_loss2 = (
                anomaly_transformer_loss(
                    x, recon, series_list, prior_list, lambda_kl=lambda_kl
                )
            )

            optim.zero_grad()
            (-loss_max2).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            losses.append(loss_min.item())
            recs.append(rec_loss.item())
            sds.append(series_loss.item())
            pds.append(prior_loss.item())

        avg_loss = float(np.mean(losses))
        print(
            f"Epoch {ep:02d}/{epochs} "
            f"loss={avg_loss:.6f} rec={np.mean(recs):.6f} "
            f"seriesKL={np.mean(sds):.6f} priorKL={np.mean(pds):.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), root / "checkpoints" / "best_metropT_anomaly_transformer.pt")
            print("  ✓ saved best")

        history["epoch"].append(ep)
        history["loss"].append(avg_loss)
        history["rec"].append(float(np.mean(recs)))
        history["seriesKL"].append(float(np.mean(sds)))
        history["priorKL"].append(float(np.mean(pds)))

    pd.DataFrame(history).to_csv(root / "logs" / "metropT_train_curve.csv", index=False)
    print("Saved train curve to logs/metropT_train_curve.csv")

    # EVAL (P/R/F1)
    model.load_state_dict(torch.load(root / "checkpoints" / "best_metropT_anomaly_transformer.pt", map_location=device))

    win_scores = compute_scores(model, test_loader, device, temperature=50.0)
    print("test window scores:", win_scores.shape)

    # 1) window->point
    T = int(meta["test_T"])
    win_size = int(meta["win_size"])
    step = int(meta["step"])
    point_scores = window_to_point_scores(win_scores, T=T, win_size=win_size, step=step)

    # 2) best F1 with point-adjust
    label = np.asarray(test_label).astype(np.int32)
    best, best_pred = best_f1_with_adjust(point_scores, label, n_thr=300)

    print(f"[BEST] thr={best['thr']:.6f}  P={best['P']:.4f}  R={best['R']:.4f}  F1={best['F1']:.4f}")

    # Debugging: analyze anomaly segments
    # def get_segments(label):
    #     segs = []
    #     in_seg = False
    #     s = 0
    #     for i, v in enumerate(label):
    #         if v == 1 and not in_seg:
    #             in_seg = True
    #             s = i
    #         if in_seg and (i == len(label) - 1 or label[i + 1] == 0):
    #             e = i
    #             segs.append((s, e))
    #             in_seg = False
    #     return segs
    #
    # label = np.asarray(test_label).astype(np.int32)
    # segs = get_segments(label)
    # print("num anomaly segments:", len(segs))
    # print("anomaly points:", int(label.sum()))
    #
    # seg_max = np.array([point_scores[s : e + 1].max() for s, e in segs])
    # seg_mean = np.array([point_scores[s : e + 1].mean() for s, e in segs])
    #
    # normal_scores = point_scores[label == 0]
    # q99 = np.quantile(normal_scores, 0.99)
    # q995 = np.quantile(normal_scores, 0.995)
    # q999 = np.quantile(normal_scores, 0.999)
    # print("normal quantiles:", q99, q995, q999)
    #
    # print("segments with max <= normal q99:", (seg_max <= q99).sum(), "/", len(segs))
    # print("segments with max <= normal q999:", (seg_max <= q999).sum(), "/", len(segs))

    # save artifacts
    np.save(root / "logs" / "metropT_test_window_scores.npy", win_scores)
    np.save(root / "logs" / "metropT_test_point_scores.npy", point_scores)
    np.save(root / "logs" / "metropT_best_pred_adjusted.npy", best_pred.astype(np.uint8))

    with open(root / "logs" / "metropT_best_metrics.txt", "w") as f:
        f.write(str(best) + "\n")

    print("Saved:")
    print(" - logs/metropT_test_window_scores.npy")
    print(" - logs/metropT_test_point_scores.npy")
    print(" - logs/metropT_best_pred_adjusted.npy")
    print(" - logs/metropT_best_metrics.txt")


if __name__ == "__main__":
    main()