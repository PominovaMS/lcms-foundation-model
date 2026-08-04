"""Tests for the deterministic probe split (eval/data.py::assign_splits).

The fixture mirrors the shape of the real abele problem in miniature: one probe
genus ("Alpha") has a single species carrying far more files than any other, which
is what lets the probe collapse onto a class that is rare at validation time.
"""

import polars as pl
import pytest

from data import assign_splits, stride

PROBE_SPLITS = ["probe_train", "probe_val"]
SPLIT_KW = {"n_probe_genera": 3, "n_ssl_top": 1, "min_species_per_genus": 2}


def _rows(genus, species, n_files, start=0):
    return [
        {"genus": genus, "organism": species, "peak_file": f"{species}_{i:03d}.mzML"}
        for i in range(start, start + n_files)
    ]


@pytest.fixture
def meta_df():
    """Four genera: one for SSL, three for the probe.

    ``Alpha one`` (12 files) is the *Escherichia coli* analogue — alphabetically
    first within its genus, so it always lands in probe_train, and 4x the size of
    every other species.
    """
    rows = []
    for i in range(4):  # largest genus -> reserved for SSL
        rows += _rows("SSLbig", f"SSLbig sp{i}", 5)
    rows += _rows("Alpha", "Alpha one", 12)  # dominant species
    rows += _rows("Alpha", "Alpha two", 3)
    for i in range(4):
        rows += _rows("Beta", f"Beta sp{i}", 3)
    for i in range(2):
        rows += _rows("Gamma", f"Gamma sp{i}", 3)
    return pl.DataFrame(rows)


def _counts(df, split):
    return dict(
        df.filter(pl.col("split") == split)
        .group_by("genus")
        .agg(pl.len().alias("n"))
        .iter_rows()
    )


def test_stride_is_evenly_spaced():
    """stride samples across the range rather than truncating to a prefix."""
    assert stride(48, 3) == [0, 24, 47]
    assert stride(3, 3) == [0, 1, 2]
    assert stride(3, 10) == [0, 1, 2]  # fewer files than the cap -> keep all
    assert stride(5, 0) == [0, 1, 2, 3, 4]  # cap disabled


def test_uncapped_split_is_imbalanced(meta_df):
    """Without a cap the dominant species skews probe_train. Regression guard."""
    df = assign_splits(meta_df, max_files_per_species=None, **SPLIT_KW)
    assert _counts(df, "probe_train")["Alpha"] == 12
    assert _counts(df, "probe_val")["Alpha"] == 3
    assert df.filter(pl.col("split") == "unused").height == 0


def test_cap_per_species_balances_the_probe(meta_df):
    """Capping files per species removes the dominant species' advantage."""
    df = assign_splits(meta_df, max_files_per_species=3, **SPLIT_KW)
    train, val = _counts(df, "probe_train"), _counts(df, "probe_val")
    assert train["Alpha"] == 3 and val["Alpha"] == 3
    # every other genus is untouched — its species were already at/below the cap
    assert train["Beta"] == 6 and val["Beta"] == 6
    assert train["Gamma"] == 3 and val["Gamma"] == 3


def test_no_species_exceeds_the_cap(meta_df):
    """The invariant the cap exists to enforce, checked per (split, species)."""
    cap = 3
    df = assign_splits(meta_df, max_files_per_species=cap, **SPLIT_KW)
    per_species = (
        df.filter(pl.col("split").is_in(PROBE_SPLITS))
        .group_by(["split", "organism"])
        .agg(pl.len().alias("n"))
    )
    assert per_species["n"].max() <= cap


def test_capped_files_are_unused_not_ssl(meta_df):
    """Capped-out files must NOT land in "train".

    Donating probe-genus files to the SSL split would leak the probe genera into
    the abele SSL corpus built by scripts/build_abele_ssl.py.
    """
    df = assign_splits(meta_df, max_files_per_species=3, **SPLIT_KW)
    unused = df.filter(pl.col("split") == "unused")
    assert unused.height == 9  # 12 Alpha-one files capped to 3
    assert set(unused["genus"].to_list()) == {"Alpha"}
    # nothing from a probe genus leaked into the SSL split
    assert set(df.filter(pl.col("split") == "train")["genus"].to_list()) == {"SSLbig"}


def test_cap_per_genus_balances_classes_exactly(meta_df):
    """The optional per-genus cap equalises the classes."""
    df = assign_splits(
        meta_df, max_files_per_species=3, max_files_per_genus=3, **SPLIT_KW
    )
    for split in PROBE_SPLITS:
        assert set(_counts(df, split).values()) == {3}


def test_ssl_split_is_never_capped(meta_df):
    """Caps apply to the probe only — SSL pretraining wants all the data."""
    df = assign_splits(meta_df, max_files_per_species=1, **SPLIT_KW)
    assert df.filter(pl.col("split") == "train").height == 20


def test_split_is_deterministic(meta_df):
    """Same input, same assignment — the docstring promises no randomness."""
    a = assign_splits(meta_df, max_files_per_species=3, **SPLIT_KW)
    b = assign_splits(meta_df, max_files_per_species=3, **SPLIT_KW)
    assert a["split"].to_list() == b["split"].to_list()
    assert a["genus_class"].to_list() == b["genus_class"].to_list()


def test_genus_class_is_stable_under_capping(meta_df):
    """Capping must not renumber the classes, or stages stop being comparable."""
    uncapped = assign_splits(meta_df, max_files_per_species=None, **SPLIT_KW)
    capped = assign_splits(meta_df, max_files_per_species=3, **SPLIT_KW)
    assert uncapped["genus_class"].to_list() == capped["genus_class"].to_list()
