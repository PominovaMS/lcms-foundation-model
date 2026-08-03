"""Compare the MS1 peak m/z (mass) distribution across datasets — no training required.

Reads MS1 spectra with the SAME preprocessing ``train.py`` uses (``filter_intensity``
then sqrt intensity scaling) and reports, per dataset "series", the distribution of peak
m/z values. Each PRIDE repository (PXD accession) is one series; abele is the reference
series every accession is compared against. Pure data property — no model, checkpoint, or GPU.

Why it matters: the model only predicts m/z bins over ``[bin_mz_min, bin_mz_max)`` (see
``config.yaml`` / ``MS1Encoder.get_mz_bins``). Peaks below ``bin_mz_min`` become ``-1``
(ignored in the loss); peaks at/above ``bin_mz_max`` all collapse into the top bin. If a
new PRIDE repo's mass distribution is shifted relative to abele — or carries a lot of mass
outside that window — that's a train/eval distribution shift worth knowing about.

Sampling: an m/z distribution is an aggregate over millions of peaks, so it stabilises on a
handful of files. Reading is parse-dominated (each mzML is fully parsed), so the file COUNT
is the cost driver: ``--limit-files`` (default 5) strided per series is the primary knob.

Usage
-----
    # one series per PRIDE repository, vs abele
    python scripts/mass_dist.py \\
        --pride-root /mnt/data/shared/lc_ms_foundation/pride_data \\
        --dataset abele /mnt/data/shared/lc_ms_foundation/abele_data/mzml \\
        --reference abele -o mass_dist.png --dump-csv mass_dist.csv

    # quick smoke test: one accession + abele, 2 files each
    python scripts/mass_dist.py \\
        --dataset PXD019483 /mnt/data/shared/lc_ms_foundation/pride_data/PXD019483/mzml \\
        --dataset abele /mnt/data/shared/lc_ms_foundation/abele_data/mzml \\
        --reference abele --limit-files 2 -o /tmp/mass_smoke.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from depthcharge.data import preprocessing, spectra_to_df

import matplotlib

matplotlib.use("Agg")  # headless server / no display
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ discovery
def find_mzml(d: Path) -> list[Path]:
    """All mzML files under ``d`` (recursive), sorted. Matches .mzml / .mzml.gz."""
    return sorted(
        p
        for p in d.rglob("*")
        if p.is_file() and p.name.lower().endswith((".mzml", ".mzml.gz"))
    )


def pride_series(
    root: str, subdir: str, accessions: list[str] | None
) -> list[tuple[str, list[Path]]]:
    """One (label, files) series per immediate subdirectory (PXD accession) of ``root``.

    mzML is looked for under ``<accession>/<subdir>`` (default ``mzml``, matching
    ``peak_stats.py`` / ``build_stage.py``), falling back to a recursive search.
    """
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"--pride-root {root} is not a directory")
    if accessions:
        subs = [root_p / a for a in accessions]
    else:
        subs = sorted(p for p in root_p.iterdir() if p.is_dir())

    series: list[tuple[str, list[Path]]] = []
    for sub in subs:
        if not sub.is_dir():
            print(f"WARNING: accession dir missing, skipping: {sub}", file=sys.stderr)
            continue
        mzdir = sub / subdir
        files = find_mzml(mzdir) if mzdir.is_dir() else find_mzml(sub)
        if files:
            series.append((sub.name, files))
        else:
            print(f"WARNING: no mzML found for {sub.name} under {sub}", file=sys.stderr)
    return series


def stride(n: int, k: int) -> list[int]:
    """``k`` indices evenly spaced across ``range(n)`` (deterministic, no RNG)."""
    if k <= 0 or n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


# ------------------------------------------------------------------ divergence
def _normalize(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else counts


def divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> dict[str, float]:
    """JSD (base 2, in [0,1]), total-variation distance, and max CDF gap (KS-style)."""
    p, q = _normalize(p_counts), _normalize(q_counts)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    tv = 0.5 * float(np.abs(p - q).sum())
    ks = float(np.abs(np.cumsum(p) - np.cumsum(q)).max())
    return {"jsd": jsd, "tv": tv, "ks": ks}


# ------------------------------------------------------------------ accumulate
def accumulate_series(
    label: str,
    files: list[Path],
    edges: np.ndarray,
    pfn: list,
    full_max: float,
    limit_files: int,
    max_spectra: int,
    intensity_weighted: bool,
) -> dict:
    """Read a strided sample of ``files`` and build fixed-edge histograms of peak m/z."""
    sampled = [files[i] for i in stride(len(files), limit_files)]
    n_grid = len(edges) - 1
    hist = np.zeros(n_grid)
    hist_w = np.zeros(n_grid) if intensity_weighted else None
    n_spectra = 0
    n_above_full = 0
    n_skipped = 0

    for i, f in enumerate(sampled):
        print(f"{label}: file {i + 1}/{len(sampled)}  {f.name}", flush=True)
        try:
            df = spectra_to_df(
                str(f), metadata_df=None, ms_level=1, preprocessing_fn=pfn,
                valid_charge=None, custom_fields=None, progress=False,
            )
        except Exception as e:  # corrupt/truncated mzML — skip, don't kill the run
            n_skipped += 1
            print(f"WARNING: skipping unreadable file {f} ({type(e).__name__}: {e})",
                  file=sys.stderr)
            continue
        if max_spectra and df.height > max_spectra:
            df = df[stride(df.height, max_spectra)]  # even-strided rows across the run
        n_spectra += df.height

        mz = df.get_column("mz_array").explode().drop_nulls().to_numpy()
        if mz.size == 0:
            continue
        hist += np.histogram(mz, bins=edges)[0]
        n_above_full += int((mz >= full_max).sum())
        if intensity_weighted:
            inten = df.get_column("intensity_array").explode().drop_nulls().to_numpy()
            hist_w += np.histogram(mz, bins=edges, weights=inten)[0]

    if n_skipped:
        print(f"{label}: skipped {n_skipped}/{len(sampled)} unreadable file(s)",
              file=sys.stderr)
    return {
        "label": label,
        "n_files": len(sampled) - n_skipped,
        "n_skipped": n_skipped,
        "n_spectra": n_spectra,
        "hist": hist,
        "hist_w": hist_w,
        "n_above_full": n_above_full,
    }


def summarize(res: dict, edges: np.ndarray, bin_mz_min: float, bin_mz_max: float) -> dict:
    """Mean/median/quantiles and model-window occupancy, derived from the histogram."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = res["hist"]
    total = hist.sum()
    n_peaks = total + res["n_above_full"]
    out = {**res, "n_peaks": int(n_peaks)}
    if total == 0:
        return {**out, "mean": float("nan"), "median": float("nan"),
                "q05": float("nan"), "q95": float("nan"),
                "pct_below": float("nan"), "pct_above": float("nan"), "pct_in": float("nan")}

    cum = np.cumsum(hist) / total

    def q(p):
        return float(centers[np.searchsorted(cum, p)])

    i_min = int(np.searchsorted(edges, bin_mz_min))
    i_max = int(np.searchsorted(edges, bin_mz_max))
    below = hist[:i_min].sum()
    in_win = hist[i_min:i_max].sum()
    above = hist[i_max:].sum() + res["n_above_full"]
    return {
        **out,
        "mean": float((centers * hist).sum() / total),
        "median": q(0.5),
        "q05": q(0.05),
        "q95": q(0.95),
        "pct_below": 100.0 * below / n_peaks,
        "pct_above": 100.0 * above / n_peaks,
        "pct_in": 100.0 * in_win / n_peaks,
    }


# ------------------------------------------------------------------ plotting
def make_plot(summaries, ref_label, bin_mz_min, bin_mz_max, edges, output, dpi):
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(11, 8), dpi=dpi)
    cmap = plt.get_cmap("tab10")

    def density(res):
        total = res["hist"].sum()
        return res["hist"] / total if total else res["hist"]

    non_ref = [s for s in summaries if s["label"] != ref_label]
    for ci, res in enumerate(non_ref):
        color = cmap(ci % 10)
        d = density(res)
        for ax in (ax_full, ax_zoom):
            ax.plot(centers, d, color=color, linewidth=1.3, alpha=0.85, label=res["label"])

    ref = next((s for s in summaries if s["label"] == ref_label), None)
    if ref is not None:
        d = density(ref)
        for ax in (ax_full, ax_zoom):
            ax.plot(centers, d, color="black", linewidth=2.5, alpha=0.95,
                    label=f"{ref_label} (reference)")

    # shade the model's prediction window on the full-range panel
    ax_full.axvspan(bin_mz_min, bin_mz_max, color="gray", alpha=0.12,
                    label="model window")
    ax_full.set_xlim(edges[0], edges[-1])
    ax_full.set_title("MS1 peak m/z distribution — full range")
    ax_zoom.set_xlim(bin_mz_min, bin_mz_max)
    ax_zoom.set_title(f"zoom to model window [{bin_mz_min:g}, {bin_mz_max:g})")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("m/z")
        ax.set_ylabel("fraction of peaks")
        ax.grid(True, alpha=0.3)
    ax_full.legend(fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}  ({len(summaries)} series)")


# ------------------------------------------------------------------ cli / main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pride-root", help="Root whose immediate subdirs are PXD accessions")
    p.add_argument("--accessions", nargs="+", metavar="PXD",
                   help="Restrict --pride-root to these accessions")
    p.add_argument("--mzml-subdir", default="mzml",
                   help="Subdir under each accession holding mzML (default: mzml)")
    p.add_argument("--dataset", action="append", nargs=2, metavar=("LABEL", "PATH"),
                   default=[], help="A single explicit series (e.g. abele). Repeatable.")
    p.add_argument("--reference", default="abele",
                   help="Series compared against + emphasised in the plot (default: abele)")
    p.add_argument("--limit-files", type=int, default=5,
                   help="Files sampled (strided) per series (0 = all). Default 5.")
    p.add_argument("--max-spectra-per-file", type=int, default=5000,
                   help="Cap spectra kept per file, evenly strided (0 = all). Default 5000.")
    p.add_argument("--all-peaks", action="store_true",
                   help="Skip filter_intensity (raw scan distribution, not the top-N)")
    p.add_argument("--intensity-weighted", action="store_true",
                   help="Also compute an intensity-weighted histogram (CSV + weighted JSD)")
    p.add_argument("--max-num-peaks", type=int, default=200, help="Match config.data.max_num_peaks")
    p.add_argument("--bin-mz-min", type=float, default=300.0, help="Match config.model.bin_mz_min")
    p.add_argument("--bin-mz-max", type=float, default=1500.0, help="Match config.model.bin_mz_max")
    p.add_argument("--full-max", type=float, default=2000.0, help="Upper m/z of the full-range grid")
    p.add_argument("--full-bin-width", type=float, default=1.0, help="Full-grid bin width (Th)")
    p.add_argument("-o", "--output", default="mass_dist.png", help="Output PNG path")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--dump-csv", help="Optional CSV of per-bin counts for later replots")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Assemble series: PRIDE accessions (one each) + explicit --dataset entries.
    series: list[tuple[str, list[Path]]] = []
    if args.pride_root:
        series += pride_series(args.pride_root, args.mzml_subdir, args.accessions)
    for label, path in args.dataset:
        files = find_mzml(Path(path))
        if files:
            series.append((label, files))
        else:
            print(f"WARNING: no mzML found for '{label}' under {path}", file=sys.stderr)
    if not series:
        raise SystemExit("No series to compare (need --pride-root and/or --dataset)")

    pfn = [preprocessing.scale_intensity(scaling="root", max_intensity=1.0)]
    if not args.all_peaks:
        pfn.insert(0, preprocessing.filter_intensity(max_num_peaks=args.max_num_peaks))

    edges = np.arange(0.0, args.full_max + args.full_bin_width, args.full_bin_width)

    summaries = []
    for label, files in series:
        res = accumulate_series(
            label, files, edges, pfn, args.full_max,
            args.limit_files, args.max_spectra_per_file, args.intensity_weighted,
        )
        summaries.append(summarize(res, edges, args.bin_mz_min, args.bin_mz_max))

    # Divergence vs the reference series (unweighted histograms).
    ref = next((s for s in summaries if s["label"] == args.reference), None)
    if ref is None:
        print(f"WARNING: reference '{args.reference}' not among series; JSD unavailable",
              file=sys.stderr)
    for s in summaries:
        if ref is not None and s["label"] != args.reference:
            s["div"] = divergence(s["hist"], ref["hist"])
            if args.intensity_weighted and s["hist_w"] is not None:
                s["div_w"] = divergence(s["hist_w"], ref["hist_w"])
        else:
            s["div"] = None

    # Order: reference first, then accessions ranked most→least shifted from it.
    summaries.sort(key=lambda s: (s["label"] != args.reference,
                                  -(s["div"]["jsd"] if s["div"] else 0.0)))

    # ---- table ----
    cols = (
        f"{'series':<16} {'files':>5} {'spectra':>9} {'peaks':>11} {'mean':>7} "
        f"{'median':>6} {'q05':>6} {'q95':>6} {'%<min':>7} {'%>=max':>7} {'%in':>6} {'JSDvsref':>8}"
    )
    print(cols)
    print("-" * len(cols))
    for s in summaries:
        jsd = "-" if s["div"] is None else f"{s['div']['jsd']:.4f}"
        print(
            f"{s['label']:<16} {s['n_files']:>5} {s['n_spectra']:>9} {s['n_peaks']:>11} "
            f"{s['mean']:>7.1f} {s['median']:>6.0f} {s['q05']:>6.0f} {s['q95']:>6.0f} "
            f"{s['pct_below']:>6.2f}% {s['pct_above']:>6.2f}% {s['pct_in']:>5.1f}% {jsd:>8}"
        )

    # ---- divergence detail vs reference ----
    if ref is not None:
        print(f"\nDivergence vs '{args.reference}' (JSD base-2 / TV / KS max-CDF-gap):")
        for s in summaries:
            if s["div"] is None:
                continue
            line = (f"  {s['label']:<16} JSD={s['div']['jsd']:.4f}  "
                    f"TV={s['div']['tv']:.4f}  KS={s['div']['ks']:.4f}")
            if args.intensity_weighted and s.get("div_w"):
                line += f"   [weighted JSD={s['div_w']['jsd']:.4f}]"
            print(line)

    # ---- plot ----
    make_plot(summaries, args.reference, args.bin_mz_min, args.bin_mz_max,
              edges, args.output, args.dpi)

    # ---- optional CSV ----
    if args.dump_csv:
        centers = 0.5 * (edges[:-1] + edges[1:])
        labels = [s["label"] for s in summaries]
        header = ["bin_center"] + [f"{lbl}_count" for lbl in labels]
        if args.intensity_weighted:
            header += [f"{lbl}_wcount" for lbl in labels]
        with open(args.dump_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in range(len(centers)):
                row = [f"{centers[r]:g}"] + [int(s["hist"][r]) for s in summaries]
                if args.intensity_weighted:
                    row += [f"{s['hist_w'][r]:g}" for s in summaries]
                w.writerow(row)
        print(f"Saved {args.dump_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
