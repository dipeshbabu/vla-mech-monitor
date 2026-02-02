from pathlib import Path
import yaml
import numpy as np
import pickle

from vla_mech_monitor.utils.paths import ensure_dir
from vla_mech_monitor.monitor.probe import fit_logistic, ProbeModel


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    out_dir = Path(cfg["out_dir"])
    feat = np.load(out_dir / "features" /
                   "train_features.npz", allow_pickle=True)

    X = feat["X"]
    y_fail = feat["y_fail"]
    y_mode = feat["y_mode"]
    modes = list(feat["modes"])

    ensure_dir(out_dir / "monitor")

    # Train a failure-within-K binary logistic as the simplest strong baseline.
    # We use mode-based model later; for closed-loop we often prefer per-mode risk directions,
    # but binary monitor + mode from argmax direction is also valid.
    clf = fit_logistic(X, y_mode, modes=modes)  # multiclass
    with open(out_dir / "monitor" / "probe.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved:", out_dir / "monitor" / "probe.pkl")


if __name__ == "__main__":
    main()
