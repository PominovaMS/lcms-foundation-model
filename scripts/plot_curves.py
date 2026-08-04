"""Plot training curves (loss / accuracy / lr) from TensorBoard event files.

The training run logs scalars via ``TensorBoardLogger`` (see ``source/train.py``):
``train_loss`` / ``train_acc_mz_bin`` are per-step, ``val_loss`` / ``val_acc_mz_bin``
are per-epoch, and ``lr`` is per optimizer step. This script reads one or more run
directories, pulls those scalars out of the event files, and writes a PNG with three
stacked panels (loss, accuracy, lr). Pass several run dirs to overlay them — handy for
comparing data-diversity stages on the same axes.

Reading the events needs only ``tensorboard`` (already a dependency of
``TensorBoardLogger``); TensorFlow is not required.

Usage
-----
    # after rsyncing logs down from the cluster, e.g. into ./tb_logs/<run>/
    python /Users/adams/Code/lcms-foundation-model/scripts/plot_curves.py stage01_lim50x4_ep50 --output /Users/adams/Projects/LC-foundation/results/stage01_lim50x4_ep50.png

    # overlay several runs
    python /Users/adams/Code/lcms-foundation-model/scripts/plot_curves.py .stage01 ./tb_logs/stage04 -o compare.png
"""

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless server / no display
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Which logged tags land in which panel. Substring match (case-insensitive) so we
# catch Lightning's ``_step`` / ``_epoch`` suffixes without enumerating them.
PANELS = [
    ("loss", ["train_loss", "val_loss"]),
    ("accuracy", ["train_acc", "val_acc"]),
    ("lr", ["lr"]),
]

# Panels that carry a train/val pair get fixed colors so val is always the same
# color across every run you overlay (and vice versa for train). A bit of alpha
# lets overlapping train/val curves both stay visible.
TRAIN_VAL_PANELS = {"loss", "accuracy"}
TRAIN_COLOR = "#4b7f52"
VAL_COLOR = "#397edc"
LINE_ALPHA = 0.75

# When multiple runs are overlaid on a train/val panel, color is spoken for
# (train vs val), so line style is used to tell runs apart instead.
RUN_LINESTYLES = ["-", "--", "-.", ":"]


def find_event_dirs(run_dir: str) -> list[str]:
    """Directories under ``run_dir`` that hold event files (incl. version_N/)."""
    dirs = []
    for root, _, files in os.walk(run_dir):
        if any(f.startswith("events.out.tfevents") for f in files):
            dirs.append(root)
    return sorted(dirs)


def load_scalars(run_dir: str) -> dict[str, tuple[list[int], list[float]]]:
    """Merge scalars from every event file under a run dir: tag -> (steps, values)."""
    merged: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    event_dirs = find_event_dirs(run_dir)
    if not event_dirs:
        raise SystemExit(f"No event files found under {run_dir}")
    for d in event_dirs:
        acc = EventAccumulator(d, size_guidance={"scalars": 0})  # 0 = load all
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            steps, vals = merged[tag]
            for ev in acc.Scalars(tag):
                steps.append(ev.step)
                vals.append(ev.value)
    # sort each tag by step (event dirs may interleave across versions)
    return {
        tag: tuple(zip(*sorted(zip(steps, vals))))
        for tag, (steps, vals) in merged.items()
        if steps
    }


def label_for(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", help="Run dir(s) holding event files")
    parser.add_argument("-o", "--output", default="curves.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--log-loss", action="store_true", help="Log-scale the loss panel's y-axis"
    )
    args = parser.parse_args()

    runs = {label_for(d): load_scalars(d) for d in args.run_dirs}

    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(10, 3.2 * len(PANELS)), dpi=args.dpi, sharex=True
    )
    if len(PANELS) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    for ax, (panel_name, wanted) in zip(axes, PANELS):
        plotted = False
        for ri, (run_name, scalars) in enumerate(runs.items()):
            run_color = cmap(ri % 10)
            run_style = RUN_LINESTYLES[ri % len(RUN_LINESTYLES)]
            for tag, (steps, vals) in scalars.items():
                low = tag.lower()
                if not any(w in low for w in wanted):
                    continue
                # ``*_loss`` and ``*_loss_mz_bin`` are logged with identical values
                # (m/z is the only active loss term); drop the redundant one.
                if panel_name == "loss" and "loss_mz_bin" in low:
                    continue
                is_val = "val" in low
                if panel_name in TRAIN_VAL_PANELS:
                    # color = train vs val (consistent across runs), style = run
                    color = VAL_COLOR if is_val else TRAIN_COLOR
                    style = run_style
                else:
                    # no train/val split here (e.g. lr) -> color = run, style = train/val
                    color = run_color
                    style = "--" if is_val else "-"
                lbl = tag if len(runs) == 1 else f"{run_name}:{tag}"
                ax.plot(
                    steps,
                    vals,
                    style,
                    color=color,
                    linewidth=1.5,
                    alpha=LINE_ALPHA,
                    label=lbl,
                )
                plotted = True
        ax.set_ylabel(panel_name)
        ax.grid(True, alpha=0.3)
        if panel_name == "loss" and args.log_loss and plotted:
            ax.set_yscale("log")
        if plotted:
            ax.legend(fontsize=7, ncol=2, loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                f"no '{panel_name}' scalars",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )

    axes[-1].set_xlabel("step")
    fig.suptitle(" vs ".join(runs) if len(runs) > 1 else next(iter(runs)))
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}  ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
