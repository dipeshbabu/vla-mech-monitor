"""CLI: compute detector coverage and precision.

Example:
  python -m libero_experiments.compute_detector_quality --run_dir runs/2026-02-16_... --manual_labels labels.jsonl
"""

from __future__ import annotations

import argparse
import os

from libero_experiments.detector_quality import compute_detector_quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing detector_events.jsonl")
    ap.add_argument("--manual_labels", default=None, help="Optional jsonl with manual labels")
    args = ap.parse_args()

    det_path = os.path.join(args.run_dir, "detector_events.jsonl")
    m = compute_detector_quality(det_path, args.manual_labels)

    print("Detector quality")
    print(f"Total episodes: {m.n_total}")
    print(f"Detected episodes: {m.n_detected}")
    print(f"Coverage: {m.coverage:.3f}")
    if args.manual_labels is not None:
        print(f"Labeled episodes: {m.n_labeled}")
        print(f"Labeled episodes detected: {m.n_labeled_detected}")
        print(f"Correct type on detected labeled: {m.n_correct}")
        print(f"Precision (on detected labeled): {None if m.precision is None else f'{m.precision:.3f}'}")


if __name__ == "__main__":
    main()
