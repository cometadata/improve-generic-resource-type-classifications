#!/usr/bin/env python
"""Evaluate the output produced by `classify.py`.

The script expects the *JSON Lines* (``.jsonl``) produced by ``classify.py`` as its
first positional argument and prints a rich textual report **and** stores helpful
graphs next to the input file.

Metrics/computations performed
-----------------------------
1. Overall accuracy
2. Per-category accuracy
3. Precision / recall / F1 using ``sklearn.metrics.classification_report``
4. Confusion matrix heat-map
5. Histograms of *model probability* for correct **vs.** incorrect predictions

Special-case handling
---------------------
1. If ``resourceTypeGeneral`` is either *Text* or *Other* we keep those as the
ground-truth label - these are the two classes we explicitly want to
re-evaluate.
2. If ``attributes.types.resourceType`` **itself** is one of the general
   resource-type categories (e.g. *Dataset*, *Image*, …) then we treat that as
   the ground-truth label **even when** ``resourceTypeGeneral`` says something
   else.  This allows us to give the model credit when it tightens an overly
   generic label such as *Collection* to the more specific *Dataset*.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import List

import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = {
    1: "Audiovisual",
    2: "Award",
    3: "Book",
    4: "BookChapter",
    5: "Collection",
    6: "ComputationalNotebook",
    7: "ConferencePaper",
    8: "ConferenceProceeding",
    9: "DataPaper",
    10: "Dataset",
    11: "Dissertation",
    12: "Event",
    13: "Image",
    14: "Instrument",
    15: "InteractiveResource",
    16: "Journal",
    17: "JournalArticle",
    18: "Model",
    19: "OutputManagementPlan",
    20: "PeerReview",
    21: "PhysicalObject",
    22: "Preprint",
    23: "Project",
    24: "Report",
    25: "Service",
    26: "Software",
    27: "Sound",
    28: "Standard",
    29: "StudyRegistration",
    30: "Text",
    31: "Workflow",
    32: "Other",
}
GENERAL_TYPES = set(CATEGORIES.values())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate classification results produced by classify.py"
    )
    parser.add_argument(
        "input_file",
        type=pathlib.Path,
        help="Path to the JSONL file created by classify.py",
    )
    parser.add_argument(
        "--out-prefix",
        type=pathlib.Path,
        default=None,
        help="Prefix for all created plots. Defaults to <INPUT_FILE.stem>_eval in the same folder.",
    )
    return parser.parse_args()


def determine_ground_truth(record: dict) -> str | None:
    """Return the effective ground-truth label for *record*.

    Implements the two special-case rules described at the top of the file.
    """
    rtg: str | None = record.get("attributes.types.resourceTypeGeneral")
    rts: str | None = record.get("attributes.types.resourceType")

    # --- Rule 1: keep Text & Other exactly as they are ---------------------
    # if rtg in {"Text", "Other"}:
    #     return rtg

    # --- Rule 2: If *resourceType* itself is a general category ------------
    if rts in GENERAL_TYPES:
        return rts  # more specific label wins

    # ----------------------------------------------------------------------
    return rtg


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def load_results(path: pathlib.Path) -> pd.DataFrame:
    """Load *path* and return a ``pandas.DataFrame`` with selected columns."""
    records: List[dict] = []
    with path.open() as fp:
        for line in tqdm(fp, total=7645706):
            if not line.strip():
                continue
            data = json.loads(line)

            pred: str | None = data.get("prediction", {}).get("category")
            prob: float | None = data.get("prediction", {}).get("probability")
            truth = determine_ground_truth(data)

            # skip rows we cannot evaluate
            if pred is None or truth is None:
                continue

            records.append(
                {
                    "true": truth,
                    "pred": pred,
                    "prob": prob,
                    "correct": pred == truth,
                }
            )

    if not records:
        raise ValueError("No evaluable lines found – check the input file.")

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_category_accuracy(df: pd.DataFrame, out_file: pathlib.Path) -> None:
    acc = (
        df.groupby("true")["correct"].mean().sort_values(ascending=False)
    )  # accuracy per category

    plt.figure(figsize=(10, 6))
    sns.barplot(x=acc.values, y=acc.index, palette="viridis")
    plt.xlabel("Accuracy")
    plt.ylabel("Category")
    plt.title("Accuracy per category")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_confusion_matrix(df: pd.DataFrame, out_file: pathlib.Path) -> None:
    labels = sorted(df["true"].unique())
    cm = confusion_matrix(df["true"], df["pred"], labels=labels, normalize="true")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Normalized count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True label")
    plt.title("Confusion matrix (row-normalized)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_probability_hist(df: pd.DataFrame, out_file: pathlib.Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df[df["correct"]]["prob"],
        color="seagreen",
        alpha=0.6,
        label="Correct",
        kde=True,
    )
    sns.histplot(
        df[~df["correct"]]["prob"],
        color="firebrick",
        alpha=0.6,
        label="Incorrect",
        kde=True,
    )
    plt.xlabel("Model probability")
    plt.ylabel("Count")
    plt.title("Probability distribution – correct vs incorrect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    out_prefix: pathlib.Path = (
        args.out_prefix
        if args.out_prefix is not None
        else args.input_file.with_suffix("").with_name(args.input_file.stem + "_eval")
    )

    def add_suffix(path: pathlib.Path, suffix: str) -> pathlib.Path:
        """Return ``path`` with *suffix* appended to the file name."""
        return path.with_name(path.name + suffix)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_file)

    # -------------------------- textual report ---------------------------
    print("===== Classification report =====")
    print(
        classification_report(
            df["true"],
            df["pred"],
            zero_division=0,
            digits=3,
        )
    )

    overall_acc = df["correct"].mean()
    print(f"Overall accuracy: {overall_acc:.3%}\n")

    # -------------------------- plots -----------------------------------
    plot_category_accuracy(df, add_suffix(out_prefix, "_category_accuracy.png"))
    plot_confusion_matrix(df, add_suffix(out_prefix, "_confusion_matrix.png"))
    if df["prob"].notna().any():
        plot_probability_hist(df, add_suffix(out_prefix, "_probability_hist.png"))

    print("Plots written to:")
    for suffix in [
        "_category_accuracy.png",
        "_confusion_matrix.png",
        "_probability_hist.png",
    ]:
        f = add_suffix(out_prefix, suffix)
        if f.exists():
            print(" •", f)


if __name__ == "__main__":
    main()
