"""
This script is based on the DSEC optical flow submission format:
https://dsec.ifi.uzh.ch/optical-flow-submission-format/

Arguments:
    --pred_dir path/to/pred_pngs    directory with predicted DSEC PNGs
    --gt_dir   path/to/gt_flo       directory with ground truth .flo files
    [--thresholds 1 2 3 5 10 20]    ANPE error thresholds for evaluation
    [--tau 0.5]                     minimum GT flow magnitude for REE
    [--output_csv results.csv]      save per frame results to CSV

"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import imageio.v3 as iio
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
pd.options.display.float_format = "{:.4f}".format

BIAS = 2**15  # 32768
SCALE = 128.0
FLOAT_EPS = 1e-6


def decode_flow_png(path: str):
    """Load DSEC format 3 channel 16bit PNG -> flow (H,W,2) float32."""
    img = iio.imread(path, plugin="PNG-FI")
    if img.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 PNG, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{path}: expected 3chan PNG, got shape {img.shape}")

    flow_x = (img[:, :, 0].astype(np.float32) - BIAS) / SCALE
    flow_y = (img[:, :, 1].astype(np.float32) - BIAS) / SCALE
    flow = np.stack((flow_x, flow_y), axis=-1)
    return flow  # (H,W,2)


def read_flo(path: str):
    """Read .flo file -> flow (H,W,2) float32."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"{path}: bad .flo magic {magic}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape((h, w, 2)).astype(np.float32)


# Metrics


@dataclass
class FrameMetrics:
    idx: str
    aepe: float
    ree: float
    aae: float
    npe: Dict[int, float]

    def as_dict(self):
        d = asdict(self)
        for thr, val in self.npe.items():
            d[f"npe>{thr}"] = val
        del d["npe"]
        return d


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_pred: np.ndarray,
    valid_gt: np.ndarray,
    *,
    thresholds: List[int],
    tau: float,
):
    """Return metrics for a single frame"""
    # Intersection mask
    mask = valid_pred & valid_gt
    if not np.any(mask):
        return None

    diff = pred - gt
    epe_map = np.linalg.norm(diff, axis=-1)
    epe_vals = epe_map[mask]

    # AEPE
    aepe = float(epe_vals.mean())

    # N‑pixel error map -> percentages
    npe = {t: float((epe_vals > t).mean()) for t in thresholds}

    # REE (ignore GT motions below tau)
    mag_gt = np.linalg.norm(gt, axis=-1)
    rel_mask = mask & (mag_gt > tau)
    ree = (
        float((epe_map[rel_mask] / (mag_gt[rel_mask] + FLOAT_EPS)).mean())
        if np.any(rel_mask)
        else float("nan")
    )

    # Angular error
    dot = (pred * gt).sum(axis=-1)
    denom = np.linalg.norm(pred, axis=-1) * mag_gt
    cos = np.clip(dot / (denom + FLOAT_EPS), -1.0, 1.0)
    ang = np.arccos(cos)
    aae = float((ang[mask].mean()) * 180.0 / np.pi)

    return aepe, ree, aae, npe


# File pairing util


def extract_index(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0]
    return (
        stem.lstrip("0") or "0"
    )  # keep as string to preserve filenames like "000003"


def gather_files(pred_dir: str, gt_dir: str):
    preds = glob.glob(os.path.join(pred_dir, "*.png"))
    gts = glob.glob(os.path.join(gt_dir, "*.flo"))

    pred_map = {extract_index(p): p for p in preds}
    gt_map = {extract_index(g): g for g in gts}

    common = sorted(pred_map.keys() & gt_map.keys())
    if not common:
        raise RuntimeError("No matching prediction/GT frame indices found")

    return [(idx, pred_map[idx], gt_map[idx]) for idx in common]


# Benchmark


def run_benchmark(
    pred_dir: str, gt_dir: str, thresholds: List[int], tau: float
):
    pairs = gather_files(pred_dir, gt_dir)
    frame_metrics: List[FrameMetrics] = []

    for idx, pred_path, gt_path in pairs:
        try:
            pred_flow = decode_flow_png(pred_path)
            gt_flow = read_flo(gt_path)
        except Exception as e:
            print(f"[ERROR] Skipping {idx}: {e}")
            continue

        if pred_flow.shape != gt_flow.shape:
            print(
                f"[ERROR] Shape mismatch {idx}: pred {pred_flow.shape}, gt {gt_flow.shape}"
            )
            continue

        # Validity Masks
        valid_pred = (
            np.isfinite(pred_flow[..., 0])
            & np.isfinite(pred_flow[..., 1])
            & (np.linalg.norm(pred_flow, axis=-1) > 0)
        )
        valid_gt = (
            np.isfinite(gt_flow[..., 0])
            & np.isfinite(gt_flow[..., 1])
            & (np.linalg.norm(gt_flow, axis=-1) > 0)
        )

        metrics = compute_metrics(
            pred_flow,
            gt_flow,
            valid_pred,
            valid_gt,
            thresholds=thresholds,
            tau=tau,
        )
        if metrics is None:
            print(
                f"[WARN] No overlapping valid pixels in frame {idx} - skipped"
            )
            continue

        aepe, ree, aae, npe = metrics
        frame_metrics.append(FrameMetrics(idx, aepe, ree, aae, npe))

    if not frame_metrics:
        raise RuntimeError("No frames with valid metrics computed")

    df = pd.DataFrame([fm.as_dict() for fm in frame_metrics]).set_index("idx")
    summary = df.mean()
    return df, summary


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate DSEC optical flow PNGs against .flo ground truth"
    )
    ap.add_argument(
        "--pred_dir",
        required=True,
        help="directory with predicted DSEC PNG files",
    )
    ap.add_argument(
        "--gt_dir", required=True, help="directory with ground truth .flo files"
    )
    ap.add_argument(
        "--thresholds",
        nargs="*",
        type=int,
        default=[1, 2, 3, 5, 10, 20],
        help="pixel thresholds for N‑pixel ANPE errors (default: 1 2 3 5 10 20)",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="minimum GT magnitude (px) included in REE calculation",
    )
    ap.add_argument("--output_csv", help="write per frame metrics to CSV")
    args = ap.parse_args()

    df, summary = run_benchmark(
        args.pred_dir, args.gt_dir, args.thresholds, args.tau
    )

    print("\nPer‑frame metrics:")
    print(df.round(4))

    print("\nOverall average:")
    print(summary.round(4))

    if args.output_csv:
        df.to_csv(args.output_csv, index=True)
        print(f"Saved per frame metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
