"""Tests for the deterministic probe split (eval/data.py::assign_splits).

The fixture mirrors the real abele shape in miniature: two large genera at a ~2:1
file ratio (the Pseudomonas/Staphylococcus analogue, 312 vs 136), where the larger
is a single heavily-replicated species and the smaller is many species with three
files each — plus smaller genera that must be excluded, and "food", which is a
sample type rather than an organism.
"""

import polars as pl
import pytest

from data import assign_splits, stride

PROBE_SPLITS = ["probe_train", "probe_val"]


def _rows(genus, species, n_files, start=0):
    return [
        {"genus": genus, "organism": species, "peak_file": f"{species}_{i:03d}.mzML"}
        for i in range(start, start + n_files)
    ]


@pytest.fixture
def meta_df():
    """Big=20 files (1 species), Mid=12 (4 species), then the tail.

    ``food`` is given more files than ``Mid`` so a test can catch it being selected
    on file count alone.
    """
    rows = []
    rows += _rows("Big", "Big one", 20)  # one replicated species
    for i in range(4):
        rows += _rows("Mid", f"Mid sp{i}", 3)  # 12 files, 4 species
    rows += _rows("food", "food sample", 15)  # excluded: a sample type
    rows += _rows("Small", "Small one", 3)
    rows += _rows("Tiny", "Tiny one", 1)
    return pl.DataFrame(rows)


def _counts(df, split, col="genus"):
    return dict(
        df.filter(pl.col("split") == split)
        .group_by(col)
        .agg(pl.len().alias("n"))
        .iter_rows()
    )


def test_stride_is_evenly_spaced():
    """stride samples across the range rather than truncating to a prefix."""
    assert stride(48, 3) == [0, 24, 47]
    assert stride(3, 3) == [0, 1, 2]
    assert stride(3, 10) == [0, 1, 2]  # fewer files than the cap -> keep all
    assert stride(5, 0) == [0, 1, 2, 3, 4]  # cap disabled


def test_selects_the_largest_labels(meta_df):
    """Top-N by file count. Regression guard for dropping n_ssl_top.

    The old split reserved the largest genera for an SSL corpus nothing builds any
    more; the biggest label must now be *first* in line, not excluded.
    """
    df = assign_splits(meta_df, n_classes=2)
    selected = set(df.filter(pl.col("label_class") >= 0)["genus"].to_list())
    assert selected == {"Big", "Mid"}


def test_food_is_never_a_class(meta_df):
    """'food' is a sample type, not an organism — excluded however many files it has."""
    df = assign_splits(meta_df, n_classes=2)
    assert "food" not in df.filter(pl.col("label_class") >= 0)["genus"].to_list()
    assert set(df.filter(pl.col("genus") == "food")["split"].to_list()) == {"unused"}


def test_every_file_of_a_selected_class_is_used(meta_df):
    """No cap, no balancing: the classes keep their natural ratio."""
    df = assign_splits(meta_df, n_classes=2)
    probe = df.filter(pl.col("split").is_in(PROBE_SPLITS))
    per_genus = dict(probe.group_by("genus").agg(pl.len().alias("n")).iter_rows())
    assert per_genus == {"Big": 20, "Mid": 12}


def test_split_is_proportional_within_each_class(meta_df):
    """val_frac applies to each class separately, so train/val match in distribution."""
    df = assign_splits(meta_df, n_classes=2, val_frac=0.3)
    train, val = _counts(df, "probe_train"), _counts(df, "probe_val")
    assert train["Big"] == 14 and val["Big"] == 6  # round(20 * 0.3) == 6
    assert train["Mid"] == 8 and val["Mid"] == 4  # round(12 * 0.3) == 4


def test_val_files_are_strided_not_contiguous(meta_df):
    """A prefix/suffix would put one acquisition block entirely on one side."""
    df = assign_splits(meta_df, n_classes=2, val_frac=0.3)
    big = df.filter(pl.col("genus") == "Big").sort("peak_file")
    splits = big["split"].to_list()
    val_idx = [i for i, s in enumerate(splits) if s == "probe_val"]
    assert val_idx == stride(20, 6)
    assert val_idx != list(range(6))  # not a prefix


def test_tiny_class_still_gets_one_file_on_each_side(meta_df):
    """The k clamp: a 3-file class must not end up with an empty val (or train)."""
    df = assign_splits(meta_df, n_classes=4)  # pulls in Small (3) and Tiny (1)
    small = df.filter(pl.col("genus") == "Small")
    assert small.filter(pl.col("split") == "probe_train").height == 2
    assert small.filter(pl.col("split") == "probe_val").height == 1

    # A single-file class has to give up one side; it keeps train, not val — a
    # class the probe never trains on is worse than one it never validates on.
    assert df.filter(pl.col("genus") == "Tiny")["split"].to_list() == ["probe_train"]


def test_class_indices_are_alphabetical_and_count_independent(meta_df):
    """Indices must not shift when file counts do, or stages stop being comparable."""
    df = assign_splits(meta_df, n_classes=2)
    mapping = dict(
        df.filter(pl.col("label_class") >= 0)
        .select(["genus", "label_class"])
        .unique()
        .iter_rows()
    )
    assert mapping == {"Big": 0, "Mid": 1}

    # Shrink the larger class below the smaller one; alphabetical order is unchanged.
    shrunk = meta_df.filter(
        (pl.col("genus") != "Big") | (pl.col("peak_file") < "Big one_005.mzML")
    )
    df2 = assign_splits(shrunk, n_classes=2)
    mapping2 = dict(
        df2.filter(pl.col("label_class") >= 0)
        .select(["genus", "label_class"])
        .unique()
        .iter_rows()
    )
    assert mapping2 == mapping


def test_unselected_files_are_unused_and_never_train(meta_df):
    """There is no SSL split any more — "train" must never appear."""
    df = assign_splits(meta_df, n_classes=2)
    assert "train" not in df["split"].to_list()
    unused = df.filter(pl.col("split") == "unused")
    assert set(unused["genus"].to_list()) == {"food", "Small", "Tiny"}
    assert set(unused["label_class"].to_list()) == {-1}


def test_split_is_deterministic(meta_df):
    """Same input, same assignment — the docstring promises no randomness."""
    a = assign_splits(meta_df, n_classes=2)
    b = assign_splits(meta_df, n_classes=2)
    assert a["split"].to_list() == b["split"].to_list()
    assert a["label_class"].to_list() == b["label_class"].to_list()


def test_species_labels_work(meta_df):
    """label_col="organism" probes species instead of genus."""
    df = assign_splits(meta_df, n_classes=2, label_col="organism")
    selected = set(df.filter(pl.col("label_class") >= 0)["organism"].to_list())
    # Big one (20) then food sample (15) — but food is excluded, so Small one (3)
    # ties with the Mid species and loses alphabetically.
    assert "Big one" in selected
    assert "food sample" not in selected
    assert len(selected) == 2


def test_label_offset_skips_the_largest_labels(meta_df):
    """--label_offset 1 probes the 2nd and 3rd largest, for a gentler imbalance.

    Big (20) is skipped; food (15) is still excluded, so the offset counts
    positions in the *eligible* ranking rather than the raw one, and the selection
    lands on Mid (12) and Small (3).
    """
    df = assign_splits(meta_df, n_classes=2, label_offset=1)
    selected = set(df.filter(pl.col("label_class") >= 0)["genus"].to_list())
    assert selected == {"Mid", "Small"}


def test_label_offset_zero_is_the_default_path(meta_df):
    """The default must be byte-for-byte what it was before the flag existed."""
    a = assign_splits(meta_df, n_classes=2)
    b = assign_splits(meta_df, n_classes=2, label_offset=0)
    assert a["split"].to_list() == b["split"].to_list()
    assert a["label_class"].to_list() == b["label_class"].to_list()


def test_too_few_labels_raises(meta_df):
    """Asking for more classes than exist must fail loudly, not silently shrink."""
    with pytest.raises(ValueError, match="need 99"):
        assign_splits(meta_df, n_classes=99)

    # The bound is offset + n_classes: 4 eligible genera, so offset 3 leaves one.
    with pytest.raises(ValueError, match="need 5"):
        assign_splits(meta_df, n_classes=2, label_offset=3)
