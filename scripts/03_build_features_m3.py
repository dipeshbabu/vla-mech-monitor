from pathlib import Path
import yaml
import numpy as np

from vla_mech_monitor.eval.aggregate import load_jsonl
from vla_mech_monitor.data.dataset import build_step_features
from vla_mech_monitor.utils.paths import ensure_dir


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    out_dir = Path(cfg["out_dir"])
    rows = load_jsonl(out_dir / "episodes_m1.jsonl")

    ds = build_step_features(
        rows,
        k_horizon=int(cfg["nearmiss"]["failure_within_k"]),
        modes=list(cfg["monitor"]["modes"]),
        site_reduce="mean",
    )
    ensure_dir(out_dir / "features")
    np.savez_compressed(
        out_dir / "features" / "train_features.npz",
        X=ds.X,
        y_fail=ds.y_fail_within_k,
        y_mode=ds.y_mode,
        modes=np.array(ds.modes, dtype=object),
    )
    print("Wrote:", out_dir / "features" / "train_features.npz")


if __name__ == "__main__":
    main()
