from pathlib import Path
import yaml

from vla_mech_monitor.eval.aggregate import load_jsonl, group_by
from vla_mech_monitor.eval.metrics import summarize_episodes
from vla_mech_monitor.utils.log import write_json
from vla_mech_monitor.utils.paths import ensure_dir


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    out_dir = Path(cfg["out_dir"])
    paper_dir = ensure_dir(out_dir / "paper_artifacts")

    m1 = load_jsonl(out_dir / "episodes_m1.jsonl")
    m5 = load_jsonl(out_dir / "episodes_m5_closed_loop.jsonl") if (out_dir /
                                                                   "episodes_m5_closed_loop.jsonl").exists() else []

    m1_sum = summarize_episodes(m1)
    write_json(paper_dir / "summary_m1.json", {
        "n": m1_sum.n,
        "success_rate": m1_sum.success_rate,
        "failure_type_counts": m1_sum.failure_type_counts,
    })

    if m5:
        m5_sum = summarize_episodes(m5)
        write_json(paper_dir / "summary_m5.json", {
            "n": m5_sum.n,
            "success_rate": m5_sum.success_rate,
            "failure_type_counts": m5_sum.failure_type_counts,
        })

    # Per-perturbation breakdown for M1
    g = group_by(m1, key="perturbation")
    # perturbation field is a dict; group_by stringifies it; that's OK for quick artifacts.

    print("Wrote paper artifacts to:", paper_dir)


if __name__ == "__main__":
    main()
