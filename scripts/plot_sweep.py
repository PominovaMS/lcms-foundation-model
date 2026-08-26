"""Plot the data-diversity scaling curve from a sweep results CSV.

Reads the CSVs produced by ``eval/probe_checkpoint.py --results_csv`` (one row per
stage) and plots downstream probe accuracy against the number of pretraining
datasets. The SLURM jobs write one CSV per run — see the note in
``run_sweep.slurm`` — so several paths may be passed and their rows are pooled.

The number of datasets per stage is taken from the trailing integer in
``run_name`` (``stage03`` -> 3); rows without one fall back to file order. Note
this misreads hyperparameter tags (``..._ep50`` -> 50, ``..._lr1e-3`` -> 3), so it
is meant for ``stageNN`` diversity sweeps rather than LR comparisons.

``--x ssl_epoch`` swaps that axis for the pretraining epoch, read from the
``ssl_epoch`` column — for a CSV that probes several checkpoints of ONE run
(``train.py --save_every_n_epochs``) instead of one checkpoint per stage.

Usage:
    python scripts/plot_sweep.py --results_csv results/sweep.csv --output sweep.png
    python scripts/plot_sweep.py --results_csv results/*.csv -o compare.png
    python scripts/plot_sweep.py --results_csv results/probe/<run>.csv \
        --x ssl_epoch -o epochs.png
"""

import argparse
import csv
import re

import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt


def stage_number(run_name: str, fallback: int) -> int:
    """Number of cumulative datasets = trailing integer in the run name."""
    m = re.search(r"(\d+)\s*$", run_name or "")
    return int(m.group(1)) if m else fallback


def load_rows(results_csvs: list[str], x: str = "stage") -> list[dict]:
    """Pool the rows of every CSV given, in the order the paths were passed.

    ``x="stage"`` puts the cumulative dataset count on the x-axis (the diversity
    sweep). ``x="ssl_epoch"`` puts the pretraining epoch there instead, read from the
    column ``probe_checkpoint.py`` fills from the checkpoint — for a CSV that probes
    several epochs of ONE run rather than one checkpoint per stage.
    """
    rows = []
    for path in results_csvs:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {', '.join(results_csvs)}")
    points = []
    for i, r in enumerate(rows, start=1):
        if x == "ssl_epoch":
            raw = (r.get("ssl_epoch") or "").strip()
            if not raw:
                raise SystemExit(
                    "--x ssl_epoch needs an ssl_epoch column; these rows predate it "
                    "(re-probe with the current eval/probe_checkpoint.py)."
                )
            xv = int(raw)
        else:
            xv = stage_number(r.get("run_name", ""), i)
        points.append(
            {
                "x": xv,
                "run_name": r.get("run_name", f"row{i}"),
                "acc": float(r["probe_val_acc"]),
                "loss": float(r.get("probe_val_loss", "nan")),
            }
        )
    points.sort(key=lambda p: p["x"])
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results_csv",
        required=True,
        nargs="+",
        help="One or more sweep results CSVs (from probe_checkpoint.py). The SLURM "
        "jobs write one per run, so a glob like results/*.csv pools them.",
    )
    parser.add_argument(
        "-o", "--output", default="sweep.png", help="Output PNG path"
    )
    parser.add_argument(
        "--x",
        choices=["stage", "ssl_epoch"],
        default="stage",
        help="What goes on the x-axis. 'stage' (default) is the cumulative dataset "
        "count from the run name — the diversity sweep. 'ssl_epoch' is the "
        "pretraining epoch, for a CSV that probes several checkpoints of one run "
        "(train.py --save_every_n_epochs).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (default: follows --x)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    points = load_rows(args.results_csv, x=args.x)
    xs = [p["x"] for p in points]
    accs = [p["acc"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)
    ax.plot(xs, accs, marker="o", linewidth=2, color="#2b6cb0")
    for p in points:
        ax.annotate(
            f"{p['acc']:.3f}",
            (p["x"], p["acc"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    if args.x == "ssl_epoch":
        ax.set_xlabel("Pretraining epoch")
        default_title = "Downstream accuracy vs. pretraining epoch"
    else:
        ax.set_xlabel("Number of PRIDE datasets in pretraining (cumulative)")
        default_title = "Downstream accuracy vs. pretraining data diversity"
    ax.set_ylabel("Probe validation accuracy")
    ax.set_title(args.title or default_title)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.margins(y=0.15)

    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output} ({len(points)} points, x={args.x})")


if __name__ == "__main__":
    main()
