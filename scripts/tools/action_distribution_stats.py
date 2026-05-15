#!/usr/bin/env python
"""Summarize CARLA expert action label distributions.

By default this script mirrors the training data path from ``CARLAData``:
it instantiates the dataset, reads ``dataset.metas``, and summarizes the
same meta files used to build training labels. A direct filesystem scan mode
is also available for quick data inspection.

  - steer
  - throttle
  - brake
  - thr_brake = throttle - float(brake)
"""

import argparse
import json
import lzma
import math
import pickle
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np


ACTION_RANGES = {
    "steer": (-1.0, 1.0),
    "throttle": (0.0, 1.0),
    "brake": (0.0, 1.0),
    "thr_brake": (-1.0, 1.0),
}


def find_meta_files_by_scan(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if root.name == "metas":
        return sorted(root.glob("*.pkl"))
    return sorted(root.rglob("metas/*.pkl"))


def decode_path(value: Any) -> Path:
    if isinstance(value, bytes):
        return Path(value.decode("utf-8"))
    if hasattr(value, "decode"):
        return Path(value.decode("utf-8"))
    return Path(str(value))


def make_config_for_dataset_mode(args: argparse.Namespace):
    # TrainingConfig consumes sys.argv as config dotlist overrides. Hide this
    # script's argparse flags from it, then set the small set of fields needed
    # to reproduce CARLAData's training sample list.
    from lead.training.config_training import TrainingConfig

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        config = TrainingConfig()
    finally:
        sys.argv = original_argv

    root = Path(args.root).expanduser()
    if args.carla_root is not None:
        config.carla_root = str(Path(args.carla_root).expanduser())
    elif root.name == "data":
        config.carla_root = str(root.parent)

    config.use_planning_decoder = True
    config.visualize_dataset = False
    config.force_rebuild_bucket = args.force_rebuild_bucket
    config.seed = args.seed
    return config


def find_meta_files_by_dataset(args: argparse.Namespace) -> list[Path]:
    from lead.data_loader.carla_dataset import CARLAData

    config = make_config_for_dataset_mode(args)
    dataset = CARLAData(
        root=str(Path(args.root).expanduser()),
        config=config,
        training_session_cache=None,
        random=not args.no_shuffle,
        build_cache=False,
        build_buckets=False,
    )
    if args.epoch != 0:
        dataset.shuffle(args.epoch)
    return [decode_path(path) for path in dataset.metas]


def find_meta_files(args: argparse.Namespace) -> list[Path]:
    if args.source == "dataset":
        return find_meta_files_by_dataset(args)
    return find_meta_files_by_scan(Path(args.root).expanduser())


def as_finite_float(value: Any) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def read_meta_pickle(path: Path) -> Any:
    try:
        with lzma.open(path, "rb") as handle:
            return pickle.load(handle)
    except lzma.LZMAError:
        with path.open("rb") as handle:
            return pickle.load(handle)


def summarize_values(values: list[float], value_range: tuple[float, float], bins: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "missing_or_invalid": 0,
            "histogram": {"bin_edges": [], "counts": []},
        }

    percentiles = np.percentile(array, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    hist_counts, hist_edges = np.histogram(array, bins=bins, range=value_range)
    range_low, range_high = value_range

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "p05": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "p50": float(percentiles[4]),
        "p75": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "p99": float(percentiles[7]),
        "max": float(percentiles[8]),
        "range": [range_low, range_high],
        "underflow": int((array < range_low).sum()),
        "overflow": int((array > range_high).sum()),
        "exact_zero": int((array == 0.0).sum()),
        "histogram": {
            "bin_edges": hist_edges.tolist(),
            "counts": hist_counts.astype(int).tolist(),
        },
    }


def load_action_values(meta_files: list[Path], max_files: int | None) -> tuple[dict[str, list[float]], dict[str, int]]:
    values = {key: [] for key in ACTION_RANGES}
    counters = {
        "files_seen": 0,
        "files_loaded": 0,
        "files_failed": 0,
        "missing_or_invalid_action": 0,
    }

    selected_files = meta_files[:max_files] if max_files is not None else meta_files
    for meta_path in tqdm(selected_files):
        counters["files_seen"] += 1
        try:
            meta = read_meta_pickle(meta_path)
        except (OSError, lzma.LZMAError, pickle.UnpicklingError, EOFError):
            counters["files_failed"] += 1
            continue

        steer = as_finite_float(meta.get("steer"))
        throttle = as_finite_float(meta.get("throttle"))
        brake = as_finite_float(meta.get("brake"))
        if steer is None or throttle is None or brake is None:
            counters["missing_or_invalid_action"] += 1
            continue

        values["steer"].append(steer)
        values["throttle"].append(throttle)
        values["brake"].append(brake)
        values["thr_brake"].append(throttle - brake)
        counters["files_loaded"] += 1

    return values, counters


def build_summary(args: argparse.Namespace) -> dict:
    root = Path(args.root).expanduser()
    meta_files = find_meta_files(args)
    values, counters = load_action_values(meta_files, args.max_files)

    summary = {
        "root": str(root),
        "source": args.source,
        "meta_files_found": len(meta_files),
        **counters,
        "stats": {
            key: summarize_values(values[key], ACTION_RANGES[key], args.bins)
            for key in ACTION_RANGES
        },
    }

    brake_array = np.asarray(values["brake"], dtype=np.float64)
    if brake_array.size > 0:
        summary["brake_counts"] = {
            "zero": int((brake_array == 0.0).sum()),
            "nonzero": int((brake_array != 0.0).sum()),
            "nonzero_ratio": float((brake_array != 0.0).mean()),
        }

    thr_brake_array = np.asarray(values["thr_brake"], dtype=np.float64)
    if thr_brake_array.size > 0:
        summary["thr_brake_sign_counts"] = {
            "negative": int((thr_brake_array < 0.0).sum()),
            "zero": int((thr_brake_array == 0.0).sum()),
            "positive": int((thr_brake_array > 0.0).sum()),
        }

    return summary


def print_text_summary(summary: dict) -> None:
    print(f"root: {summary['root']}")
    print(f"meta_files_found: {summary['meta_files_found']}")
    print(f"files_loaded: {summary['files_loaded']}")
    print(f"files_failed: {summary['files_failed']}")
    print(f"missing_or_invalid_action: {summary['missing_or_invalid_action']}")
    print()

    for key, stats in summary["stats"].items():
        print(f"[{key}]")
        if stats["count"] == 0:
            print("  count: 0")
            continue
        fields = [
            "count",
            "mean",
            "std",
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
            "underflow",
            "overflow",
            "exact_zero",
        ]
        for field in fields:
            value = stats[field]
            if isinstance(value, float):
                print(f"  {field}: {value:.6f}")
            else:
                print(f"  {field}: {value}")
        print()

    if "brake_counts" in summary:
        print("[brake_counts]")
        for key, value in summary["brake_counts"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print()

    if "thr_brake_sign_counts" in summary:
        print("[thr_brake_sign_counts]")
        for key, value in summary["thr_brake_sign_counts"].items():
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize steer/throttle/brake/thr_brake distributions from CARLA metas.",
    )
    parser.add_argument(
        "root",
        type=str,
        help=(
            "CARLAData root in dataset mode, usually config.carla_data. "
            "In scan mode, this may also be a route metas directory or one .pkl file."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["dataset", "scan"],
        default="dataset",
        help=(
            "dataset: instantiate CARLAData and use dataset.metas, matching training. "
            "scan: recursively scan for metas/*.pkl."
        ),
    )
    parser.add_argument(
        "--carla-root",
        type=str,
        default=None,
        help=(
            "Optional config.carla_root override for dataset mode. If root ends with "
            "'data', the parent directory is used automatically."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch used for CARLAData.shuffle(epoch) in dataset mode.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable CARLAData sample shuffling in dataset mode.",
    )
    parser.add_argument(
        "--force-rebuild-bucket",
        action="store_true",
        help="Rebuild bucket collection instead of using the cached one in dataset mode.",
    )
    parser.add_argument(
        "--carla-num-samples",
        type=int,
        default=0,
        help="Override config.carla_num_samples in dataset mode. 0 means use all samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Seed used by CARLAData.shuffle in dataset mode.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins for each action value.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of meta files to scan.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write the full summary, including histogram counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    print_text_summary(summary)

    if args.json is not None:
        output_path = Path(args.json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote json: {output_path}")


if __name__ == "__main__":
    main()
