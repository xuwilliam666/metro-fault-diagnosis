import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def parse_args():
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Run decision-level integration demo for Metro monitoring")
    parser.add_argument(
        "--anomaly_scores",
        type=Path,
        default=root / "logs" / "metropT_test_point_scores.npy",
        help="Point-level anomaly score file (.npy or .csv)",
    )
    parser.add_argument(
        "--classifier_confidence",
        type=Path,
        default=None,
        help="Optional classifier confidence file (.npy or .csv). If omitted, a transparent demo confidence is derived.",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=root / "data" / "processed" / "MetroPT" / "test_label.npy",
        help="Optional ground-truth anomaly label file (.npy or .csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root / "logs" / "integration_demo",
        help="Directory for CSV, figure, and summary outputs",
    )
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=None,
        help="Threshold T for anomaly score. If omitted, uses the 0.95 quantile of anomaly scores.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.60,
        help="Threshold P for classifier confidence in fusion logic",
    )
    parser.add_argument(
        "--demo_rise_window",
        type=int,
        default=3000,
        help="Rise window for demo confidence generation when no classifier confidence file is given",
    )
    parser.add_argument(
        "--min_warning_len",
        type=int,
        default=200,
        help="Minimum duration for warning segments in the fused state sequence",
    )
    parser.add_argument(
        "--min_fault_len",
        type=int,
        default=400,
        help="Minimum duration for fault segments in the fused state sequence",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Integrated Metro Monitoring Decision-Level Fusion Demo",
        help="Figure title",
    )
    parser.add_argument(
        "--focus_window",
        type=int,
        default=40000,
        help="Window length for the report-friendly focused integration figure",
    )
    return parser.parse_args()


def load_1d_series(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            arr = df.iloc[:, 0].to_numpy()
        else:
            preferred = [c for c in df.columns if c.lower() in {"score", "confidence", "label", "value"}]
            if preferred:
                arr = df[preferred[0]].to_numpy()
            else:
                arr = df.iloc[:, -1].to_numpy()
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")

    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"empty or invalid 1D series in: {path}")
    return arr


def align_series_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == target_len:
        return arr
    if arr.size == 1:
        return np.full(target_len, float(arr[0]), dtype=np.float64)

    old_x = np.linspace(0.0, 1.0, num=arr.size)
    new_x = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(new_x, old_x, arr)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return np.asarray(arr, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="same")


def derive_demo_classifier_confidence(
    anomaly_scores: np.ndarray,
    anomaly_threshold: float,
    rise_window: int = 3000,
) -> np.ndarray:
    """
    Transparent fallback for the final system demo when no real aligned classifier
    confidence sequence exists. Confidence rises only after sustained anomalous
    behaviour, so the integrated system naturally passes through a warning state.
    """
    scores = np.asarray(anomaly_scores, dtype=np.float64)
    score_norm = normalize_01(scores)
    anomaly_mask = scores >= float(anomaly_threshold)

    streak = np.zeros_like(score_norm)
    run = 0
    for i, is_anom in enumerate(anomaly_mask):
        run = run + 1 if is_anom else 0
        streak[i] = run

    rise_window = max(int(rise_window), 1)
    persistence = np.clip(streak / float(rise_window), 0.0, 1.0)

    # Keep confidence low during early anomalous behaviour so the integrated
    # system passes through a visible warning stage before escalating to fault.
    slow_persistence = persistence ** 1.8
    confidence = 0.10 + 0.82 * (0.25 * score_norm + 0.75 * slow_persistence)
    confidence = moving_average(confidence, window=max(15, rise_window // 30))
    return np.clip(confidence, 0.0, 1.0)


def fuse_states(anomaly_scores: np.ndarray, classifier_confidence: np.ndarray, thr_anom: float, thr_conf: float):
    states = np.full(anomaly_scores.shape[0], "Normal", dtype=object)
    high_anom = anomaly_scores >= thr_anom
    states[high_anom] = "Warning"
    states[high_anom & (classifier_confidence >= thr_conf)] = "Fault"
    return states


def _run_length_encode(states):
    if len(states) == 0:
        return []

    runs = []
    start = 0
    current = states[0]
    for i in range(1, len(states)):
        if states[i] != current:
            runs.append((start, i, current))
            start = i
            current = states[i]
    runs.append((start, len(states), current))
    return runs


def smooth_states(states, min_warning_len: int, min_fault_len: int):
    states = np.asarray(states, dtype=object).copy()
    if states.size == 0:
        return states

    min_len = {"Warning": int(min_warning_len), "Fault": int(min_fault_len)}

    changed = True
    while changed:
        changed = False
        runs = _run_length_encode(states)
        for idx, (start, end, state) in enumerate(runs):
            if state == "Normal":
                continue

            seg_len = end - start
            threshold = min_len[state]
            if seg_len >= threshold:
                continue

            left_state = runs[idx - 1][2] if idx > 0 else None
            right_state = runs[idx + 1][2] if idx + 1 < len(runs) else None

            replacement = "Normal"
            if left_state == right_state and left_state is not None:
                replacement = left_state
            elif right_state is not None and right_state != state:
                replacement = right_state
            elif left_state is not None and left_state != state:
                replacement = left_state

            states[start:end] = replacement
            changed = True
            break

    return states


def state_to_code(states):
    mapping = {"Normal": 0, "Warning": 1, "Fault": 2}
    return np.asarray([mapping[s] for s in states], dtype=np.int64)


def find_first_index(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    return None if idx.size == 0 else int(idx[0])


def find_first_positive_segment(label: np.ndarray):
    label = np.asarray(label, dtype=np.int64)
    idx = np.flatnonzero(label > 0)
    if idx.size == 0:
        return None
    start = int(idx[0])
    end = start
    while end + 1 < label.size and label[end + 1] > 0:
        end += 1
    return start, end


def transition_summary(states):
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            transitions.append({"time_index": int(i), "from": str(states[i - 1]), "to": str(states[i])})
    return transitions


def build_summary(anomaly_scores, classifier_confidence, states, thr_anom, thr_conf, ground_truth, confidence_source):
    state_codes = state_to_code(states)
    unique, counts = np.unique(states, return_counts=True)
    state_counts = {str(k): int(v) for k, v in zip(unique, counts)}

    first_warning = find_first_index(states == "Warning")
    first_fault = find_first_index(states == "Fault")
    first_anomaly_crossing = find_first_index(anomaly_scores >= thr_anom)

    summary = {
        "num_points": int(len(states)),
        "anomaly_threshold": float(thr_anom),
        "confidence_threshold": float(thr_conf),
        "confidence_source": confidence_source,
        "state_counts": state_counts,
        "state_ratio": {k: float(v) / float(len(states)) for k, v in state_counts.items()},
        "first_anomaly_threshold_crossing": first_anomaly_crossing,
        "first_warning_time": first_warning,
        "first_fault_time": first_fault,
        "num_transitions": int(np.sum(np.diff(state_codes) != 0)),
        "transitions": transition_summary(states),
        "anomaly_score_min": float(np.min(anomaly_scores)),
        "anomaly_score_max": float(np.max(anomaly_scores)),
        "classifier_confidence_min": float(np.min(classifier_confidence)),
        "classifier_confidence_max": float(np.max(classifier_confidence)),
    }

    if first_warning is not None and first_fault is not None:
        summary["warning_to_fault_delay"] = int(first_fault - first_warning)
    else:
        summary["warning_to_fault_delay"] = None

    if ground_truth is not None:
        gt_seg = find_first_positive_segment(ground_truth)
        summary["ground_truth_positive_points"] = int(np.sum(ground_truth > 0))
        summary["ground_truth_first_segment"] = None if gt_seg is None else {"start": gt_seg[0], "end": gt_seg[1]}

        if gt_seg is not None:
            gt_start = gt_seg[0]
            summary["warning_before_ground_truth"] = bool(first_warning is not None and first_warning < gt_start)
            summary["fault_before_ground_truth"] = bool(first_fault is not None and first_fault < gt_start)
            summary["warning_delay_to_ground_truth_start"] = None if first_warning is None else int(first_warning - gt_start)
            summary["fault_delay_to_ground_truth_start"] = None if first_fault is None else int(first_fault - gt_start)
        else:
            summary["warning_before_ground_truth"] = None
            summary["fault_before_ground_truth"] = None
            summary["warning_delay_to_ground_truth_start"] = None
            summary["fault_delay_to_ground_truth_start"] = None

    return summary


def choose_focus_range(states: np.ndarray, ground_truth: np.ndarray | None, focus_window: int):
    n = len(states)
    focus_window = min(max(int(focus_window), 1000), n)

    first_warning = find_first_index(states == "Warning")
    first_fault = find_first_index(states == "Fault")
    center = None

    if first_fault is not None:
        center = first_fault
    elif first_warning is not None:
        center = first_warning
    elif ground_truth is not None and np.any(ground_truth > 0):
        seg = find_first_positive_segment(ground_truth)
        if seg is not None:
            center = seg[0]

    if center is None:
        center = n // 2

    start = max(0, int(center - focus_window // 3))
    end = min(n, start + focus_window)
    start = max(0, end - focus_window)
    return start, end


def add_state_background(ax, x, states_window):
    colors = {"Normal": "#dcecf8", "Warning": "#fff1bf", "Fault": "#f7c9cf"}
    runs = _run_length_encode(states_window)
    for start, end, state in runs:
        x0 = x[start]
        x1 = x[end - 1] if end - 1 < len(x) else x[-1]
        ax.axvspan(x0, x1, color=colors[state], alpha=0.28, linewidth=0)


def plot_focused_integration_figure(
    anomaly_scores: np.ndarray,
    classifier_confidence: np.ndarray,
    states: np.ndarray,
    thr_anom: float,
    thr_conf: float,
    out_path_png: Path,
    out_path_pdf: Path,
    title: str,
    ground_truth: np.ndarray | None = None,
    focus_window: int = 40000,
):
    start, end = choose_focus_range(states=states, ground_truth=ground_truth, focus_window=focus_window)
    xs = np.arange(start, end)
    anomaly = anomaly_scores[start:end]
    conf = classifier_confidence[start:end]
    states_w = states[start:end]
    gt = None if ground_truth is None else ground_truth[start:end]

    fig, axes = plt.subplots(
        3, 1, figsize=(8.3, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 0.62]},
    )

    # Top panel
    add_state_background(axes[0], xs, states_w)
    axes[0].plot(xs, anomaly, color="#2f5d8a", linewidth=1.2)
    axes[0].axhline(thr_anom, color="#444444", linestyle="--", linewidth=1.0, label="Anomaly threshold")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)

    # Middle panel
    add_state_background(axes[1], xs, states_w)
    axes[1].plot(xs, conf, color="#c97828", linewidth=1.2)
    axes[1].axhline(thr_conf, color="#444444", linestyle="--", linewidth=1.0, label="Confidence threshold")
    axes[1].set_ylabel("Classifier Confidence")
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)

    # Bottom panel
    colors = {"Normal": "#dcecf8", "Warning": "#fff1bf", "Fault": "#f7c9cf"}
    runs = _run_length_encode(states_w)
    for s0, s1, state in runs:
        axes[2].axvspan(xs[s0], xs[s1 - 1], ymin=0.28, ymax=0.72, color=colors[state], linewidth=0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_yticks([])
    axes[2].set_ylabel("State")
    axes[2].set_xlabel("Time Index")

    legend_items = [
        Patch(facecolor=colors["Normal"], edgecolor="none", label="Normal"),
        Patch(facecolor=colors["Warning"], edgecolor="none", label="Warning"),
        Patch(facecolor=colors["Fault"], edgecolor="none", label="Fault"),
    ]
    if gt is not None and np.any(gt > 0):
        segs = np.flatnonzero(np.diff(np.pad((gt > 0).astype(np.int8), (1, 1))) != 0)
        for s0, s1 in zip(segs[::2], segs[1::2]):
            for ax in axes:
                ax.axvspan(xs[s0], xs[s1 - 1], color="#6b7280", alpha=0.10, linewidth=0)
        legend_items.append(Patch(facecolor="#6b7280", edgecolor="none", alpha=0.10, label="Ground Truth"))

    axes[2].legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(legend_items)), frameon=False, fontsize=9)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#4b5563")

    fig.suptitle(f"{title} (Focused View)", y=0.975, fontsize=13)
    fig.tight_layout(rect=[0.035, 0.04, 0.985, 0.93])
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_integration_figure(
    anomaly_scores: np.ndarray,
    classifier_confidence: np.ndarray,
    states: np.ndarray,
    thr_anom: float,
    thr_conf: float,
    out_path_png: Path,
    out_path_pdf: Path,
    title: str,
    ground_truth: np.ndarray | None = None,
):
    x = np.arange(len(anomaly_scores))
    state_codes = state_to_code(states)
    first_warning = find_first_index(states == "Warning")
    first_fault = find_first_index(states == "Fault")

    fig, axes = plt.subplots(
        3, 1, figsize=(8.3, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 0.9]},
    )

    panels = [
        ("Anomaly Score", anomaly_scores, thr_anom, "#2f5d8a"),
        ("Classifier Confidence", classifier_confidence, thr_conf, "#5b5b5b"),
    ]
    for ax, (ylabel, series, threshold, color) in zip(axes[:2], panels):
        ax.plot(x, series, color=color, linewidth=1.2)
        ax.axhline(threshold, color="#444444", linestyle="--", linewidth=1.0, label="threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

        if first_warning is not None:
            ax.axvline(first_warning, color="#d4a017", linestyle="--", linewidth=1.0, alpha=0.95)
        if first_fault is not None:
            ax.axvline(first_fault, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.95)

        if ground_truth is not None and np.any(ground_truth > 0):
            segments = np.flatnonzero(np.diff(np.pad((ground_truth > 0).astype(np.int8), (1, 1))) != 0)
            for start, end in zip(segments[::2], segments[1::2]):
                ax.axvspan(start, end, color="#d9d9d9", alpha=0.35, linewidth=0)

    ax = axes[2]
    state_colors = {"Normal": "#dcecf8", "Warning": "#fff1bf", "Fault": "#f7c9cf"}
    runs = _run_length_encode(states)
    for start, end, state in runs:
        ax.axvspan(start, end, ymin=0.18, ymax=0.82, color=state_colors[state], linewidth=0)

    if ground_truth is not None and np.any(ground_truth > 0):
        segments = np.flatnonzero(np.diff(np.pad((ground_truth > 0).astype(np.int8), (1, 1))) != 0)
        for start, end in zip(segments[::2], segments[1::2]):
            ax.axvspan(start, end, ymin=0.05, ymax=0.95, color="#6b7280", alpha=0.10, linewidth=0)

    if first_warning is not None:
        ax.axvline(first_warning, color="#d4a017", linestyle="--", linewidth=1.1)
        ax.text(first_warning, 0.96, "First Warning", color="#9a7300", fontsize=9, ha="left", va="bottom")
    if first_fault is not None:
        ax.axvline(first_fault, color="#c44e52", linestyle="--", linewidth=1.1)
        ax.text(first_fault, 0.80, "First Fault", color="#9f2f34", fontsize=9, ha="left", va="bottom")

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("System State")
    ax.set_xlabel("Time Index")
    ax.set_xticks(np.linspace(0, max(len(x) - 1, 1), num=6, dtype=int))
    ax.grid(False)

    legend_items = [
        Patch(facecolor=state_colors["Normal"], edgecolor="none", label="Normal"),
        Patch(facecolor=state_colors["Warning"], edgecolor="none", label="Warning"),
        Patch(facecolor=state_colors["Fault"], edgecolor="none", label="Fault"),
    ]
    if ground_truth is not None and np.any(ground_truth > 0):
        legend_items.append(Patch(facecolor="#6b7280", edgecolor="none", alpha=0.10, label="Ground-Truth Anomaly"))
    ax.legend(handles=legend_items, loc="upper left", ncol=min(4, len(legend_items)), frameon=False, fontsize=9)

    fig.suptitle(title, y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.965])
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_scores = load_1d_series(args.anomaly_scores)
    thr_anom = float(np.quantile(anomaly_scores, 0.95)) if args.anomaly_threshold is None else float(args.anomaly_threshold)

    ground_truth = None
    if args.ground_truth is not None and Path(args.ground_truth).exists():
        ground_truth = align_series_to_length(load_1d_series(args.ground_truth), target_len=len(anomaly_scores))
        ground_truth = (ground_truth > 0).astype(np.int64)

    if args.classifier_confidence is not None:
        classifier_confidence = align_series_to_length(load_1d_series(args.classifier_confidence), target_len=len(anomaly_scores))
        classifier_confidence = np.clip(classifier_confidence, 0.0, 1.0)
        confidence_source = "loaded"
    else:
        classifier_confidence = derive_demo_classifier_confidence(
            anomaly_scores=anomaly_scores,
            anomaly_threshold=thr_anom,
            rise_window=args.demo_rise_window,
        )
        confidence_source = "derived_demo_from_anomaly_score"

    states = fuse_states(
        anomaly_scores=anomaly_scores,
        classifier_confidence=classifier_confidence,
        thr_anom=thr_anom,
        thr_conf=float(args.confidence_threshold),
    )
    states = smooth_states(
        states=states,
        min_warning_len=args.min_warning_len,
        min_fault_len=args.min_fault_len,
    )

    df = pd.DataFrame(
        {
            "time_index": np.arange(len(anomaly_scores), dtype=np.int64),
            "anomaly_score": anomaly_scores,
            "classifier_confidence": classifier_confidence,
            "state": states,
        }
    )
    if ground_truth is not None:
        df["ground_truth_anomaly"] = ground_truth.astype(np.int64)

    csv_path = args.output_dir / "integration_state_sequence.csv"
    fig_path_png = args.output_dir / "integration_result.png"
    fig_path_pdf = args.output_dir / "integration_result.pdf"
    overview_png = args.output_dir / "integration_overview.png"
    overview_pdf = args.output_dir / "integration_overview.pdf"
    summary_path = args.output_dir / "integration_summary.json"

    df.to_csv(csv_path, index=False)
    plot_integration_figure(
        anomaly_scores=anomaly_scores,
        classifier_confidence=classifier_confidence,
        states=states,
        thr_anom=thr_anom,
        thr_conf=float(args.confidence_threshold),
        out_path_png=overview_png,
        out_path_pdf=overview_pdf,
        title=f"{args.title} (Overview)",
        ground_truth=ground_truth,
    )
    plot_focused_integration_figure(
        anomaly_scores=anomaly_scores,
        classifier_confidence=classifier_confidence,
        states=states,
        thr_anom=thr_anom,
        thr_conf=float(args.confidence_threshold),
        out_path_png=fig_path_png,
        out_path_pdf=fig_path_pdf,
        title=args.title,
        ground_truth=ground_truth,
        focus_window=args.focus_window,
    )

    summary = build_summary(
        anomaly_scores=anomaly_scores,
        classifier_confidence=classifier_confidence,
        states=states,
        thr_anom=thr_anom,
        thr_conf=float(args.confidence_threshold),
        ground_truth=ground_truth,
        confidence_source=confidence_source,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved integration outputs:")
    print(" -", csv_path)
    print(" -", fig_path_png)
    print(" -", fig_path_pdf)
    print(" -", overview_png)
    print(" -", overview_pdf)
    print(" -", summary_path)
    print("confidence_source:", confidence_source)
    print("anomaly_threshold:", thr_anom)
    print("confidence_threshold:", float(args.confidence_threshold))


if __name__ == "__main__":
    main()
