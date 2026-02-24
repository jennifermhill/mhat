import csv
from pathlib import Path

import numpy as np


def agglomerate(fragments: np.ndarray, merge_history: Path, threshold: float):
    """Takes fragments and a merge history, and merges the fragments up until
    the threshold. Returns a new numpy array that is a "final" segmentation.

    Args:
        fragments (np.ndarray): Fragments that have already been run through
            waterz
        merge_history (Path): Result of running waterz on fragments with pretty
            high threshold
        threshold (float): The cost threshold to merge until. If it is bigger
            than the largest in the merge history, nothing more will merge.

    Returns:
        np.ndarray: A segmentation coming from merging the fragments according
            to the merge history
    """
    with open(merge_history) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            # Support both "cost" (new) and "score" (old) column names
            cost = float(row.get("cost", row.get("score", 0.0)))
            if cost <= threshold:
                fragments[fragments == a] = c
                fragments[fragments == b] = c

    return fragments
