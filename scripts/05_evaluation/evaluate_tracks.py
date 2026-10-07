import argparse
import json
from pathlib import Path

import geff
import toml

from mhat.dataset import Dataset, absolute_path, require_dir
from mhat.evaluation.diagnostics import run_diagnostics
from mhat.evaluation.evaluate_tracking import compute_ctc_seg, evaluate_tracking
from mhat.evaluation.linajea_metrics import (
    MATCHING_THRESHOLD_UM,
    crop_time,
    evaluate as evaluate_linajea,
    load_geff_tracks,
)
from mhat.utils import spatial_axis_names


def report_axis_consistency(gt_tracks_path, pred_axes):
    """Report -- and only report -- a GT/prediction axis-name disagreement.

    A ground truth converted under different axes than the current prediction
    would have its coordinates matched against the wrong axes. Worth saying
    early and in context.

    This deliberately does not act. Hand-annotated ground truth (NC281,
    primary_nk_cells, ...) can be the ONLY copy of those annotations, so a
    disagreement is something to repair in place, never something to resolve
    by deleting the store.
    """
    if pred_axes is None:
        return
    gt_axes = geff.GeffMetadata.read(gt_tracks_path).axes
    if gt_axes is None:
        return
    gt_names = spatial_axis_names(gt_axes)
    pred_names = spatial_axis_names(pred_axes)
    if gt_names != pred_names:
        print(
            f"WARNING: ground truth at {gt_tracks_path} has spatial axes "
            f"{gt_names} but the prediction has {pred_names}. Matching them "
            f"would compare different coordinates. Repair the ground truth "
            f"store in place -- do NOT delete it, it may be the only copy of "
            f"those annotations."
        )


# linajea is not a traccuracy metric, so it is dispatched separately below and
# must be kept out of the list handed to evaluate_tracking.
LINAJEA_METRIC = "linajea"


def run_linajea_metrics(config, pred_data_dir):
    """Score the prediction using the linajea error definitions.

    Separate from ``evaluate_tracking`` because linajea is not a traccuracy
    metric: it does its own nearest-neighbor edge matching within
    ``matching_threshold`` micrometers and reports error *counts* normalized by
    the number of GT edges, which is the form the published linajea/TGMM
    baselines are quoted in.

    The DRO ground truth ships as two independently annotated sides, so each is
    scored on its own -- matching how ``linajea_baselines_t261-310.json``
    reports ``linajea_side_1`` and ``linajea_side_2`` separately.

    Each side is a path in ``linajea_gt`` (its ``correct_tracks.zarr``); results
    are keyed by the side's directory name, e.g. ``gt_side_1``, as before.
    """
    gt_sides = [Path(p) for p in config["linajea_gt"]]
    threshold = config.get("linajea_match_threshold", MATCHING_THRESHOLD_UM)
    sparse = config.get("linajea_sparse", True)
    t_min = config.get("linajea_t_min", None)
    t_max = config.get("linajea_t_max", None)

    rec_full = load_geff_tracks(pred_data_dir / "pred_tracks.zarr")

    results = {}
    for gt_path in gt_sides:
        side = gt_path.parent.name
        gt = load_geff_tracks(require_dir(gt_path, "linajea_gt entry"))
        rec = rec_full
        # Both graphs must be cropped with the same bounds, or an edge
        # straddling a boundary is dropped from one side only and scores as a
        # spurious FN. Left unset when the GT geff is already written on the
        # same frame range as the tracked crop (the DRO gt_side_* geffs are).
        if t_min is not None or t_max is not None:
            lo = t_min if t_min is not None else -(2 ** 31)
            hi = t_max if t_max is not None else 2 ** 31
            gt = crop_time(gt, lo, hi)
            rec = crop_time(rec_full, lo, hi)
        report = evaluate_linajea(
            gt, rec, matching_threshold=threshold, sparse=sparse
        )
        results[side] = report.as_dict()
        norm = report.normalized()
        print(
            f"linajea {side}: sum_errors={report.sum_errors} "
            f"(normalized {norm['sum']:.4f}) | fn_edges={report.fn_edges}, "
            f"identity_switches={report.identity_switches}, "
            f"fp_divisions={report.fp_divisions}, "
            f"fn_divisions={report.fn_divisions}/{report.gt_divisions}"
        )
    return results


def gt_paths(config):
    """The ground-truth paths from the config, each checked to exist if given.

    Returns:
        (gt_tracks, gt_seg, ctc_seg_dir); the last two may be None.
    """
    gt_tracks = Path(config["gt_tracks"])
    if not gt_tracks.is_dir():
        raise FileNotFoundError(
            f"Ground-truth tracks {gt_tracks} are missing. For Cell Tracking "
            f"Challenge ground truth, create them first with "
            f"scripts/05_evaluation/convert_ctc_gt.py."
        )
    gt_seg = config.get("gt_seg")
    gt_seg = require_dir(Path(gt_seg), "Ground-truth segmentation") if gt_seg else None
    ctc_seg_dir = config.get("ctc_seg_dir")
    ctc_seg_dir = (
        require_dir(Path(ctc_seg_dir), "CTC SEG folder") if ctc_seg_dir else None
    )
    return gt_tracks, gt_seg, ctc_seg_dir


def run_evaluation(config, gt_tracks, gt_seg, ctc_seg_dir, pred_data_dir):

    pred_tracks_path = pred_data_dir / "pred_tracks.zarr"
    pred_axes = geff.GeffMetadata.read(pred_tracks_path).axes

    report_axis_consistency(gt_tracks, pred_axes)

    requested_metrics = list(config.get("metrics", []))
    run_linajea = LINAJEA_METRIC in requested_metrics
    # evaluate_tracking validates every name against its traccuracy
    # metrics_dict and raises on anything unknown, so strip linajea first.
    traccuracy_metrics = [m for m in requested_metrics if m != LINAJEA_METRIC]

    track_metrics = {}
    matched = None
    # When linajea is the only metric requested there is nothing for traccuracy
    # to do, and calling it anyway would silently fall back to its
    # ["basic", "track_overlap"] default on an empty list.
    if run_linajea and not traccuracy_metrics:
        print("Only linajea metrics requested; skipping traccuracy evaluation.")
    else:
        tc_config = (
            {**config, "metrics": traccuracy_metrics} if run_linajea else config
        )
        results, matched = evaluate_tracking(
            tc_config,
            gt_tracks,
            gt_seg,
            pred_data_dir,
            return_matched=True,
        )

        for metric in results:
            metric_name = metric['metric']['name']
            metric_results = metric['results']
            track_metrics[metric_name] = metric_results

    # CTC SEG uses the sparse `SEG` ground-truth folder (pixel-accurate,
    # per-slice), not the coarse `TRA` markers used for TRA/DET/LNK matching.
    if "ctc" in config.get("metrics", []):
        # `seg_track_result` lets SEG be scored against a different tracking
        # result than TRA/DET/LNK. That is needed whenever `track_result` has
        # been pruned to the CTC-evaluated lineages (see ctc_seed_prune): the
        # SEG reference annotates cells the tracking benchmark excludes, and the
        # segmentation benchmark filters extras itself, so SEG belongs on the
        # unpruned run. Left unset, SEG uses `track_result` as before.
        # Not `seg_result` -- in a *tracking* config that name is the
        # segmentation-stage uid (run_tracking.py, fn_analysis.py).
        seg_track_result = config.get("seg_track_result", None)
        seg_dir = pred_data_dir
        if seg_track_result is not None:
            seg_dir = pred_data_dir.parent / seg_track_result
            print(f"SEG measured on {seg_dir} (seg_track_result)")
        pred_seg_path = seg_dir / "pred_seg.zarr"
        if ctc_seg_dir is not None and pred_seg_path.exists():
            seg_score = compute_ctc_seg(ctc_seg_dir, pred_seg_path)
            if seg_score is not None:
                track_metrics.setdefault("CTCMetrics", {})["SEG"] = seg_score
        else:
            print(f"Skipping SEG: no ctc_seg_dir given, or no {pred_seg_path}")

    if run_linajea:
        track_metrics["LinajeaMetrics"] = run_linajea_metrics(config, pred_data_dir)

    return track_metrics, matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    ds = Dataset.from_config(config)
    track_uid = config["track_result"]
    pred_data_dir = require_dir(ds.track_dir(track_uid), "Tracking run")
    gt_tracks, gt_seg, ctc_seg_dir = gt_paths(config)

    track_metrics, matched = run_evaluation(
        config, gt_tracks, gt_seg, ctc_seg_dir, pred_data_dir
    )

    # Evaluations live inside the run they score, one directory each, named
    # after the ground truth and the metrics so none overwrites another.
    gt_label = config.get("eval_name") or absolute_path(gt_tracks).parent.name
    metrics = config.get("metrics", ["basic", "track_overlap"])
    output_dir = ds.new_eval_dir(track_uid, gt_label, metrics)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "track_metrics.json", 'w') as f:
        json.dump(track_metrics, f)

    # Save the eval config used for this run alongside the results for provenance
    with open(output_dir / "eval_config.toml", "w") as config_file:
        toml.dump(config, config_file)

    # FN edge / FN node / GT drift diagnostics, reusing the matching computed above
    run_diagnostics(config, gt_tracks.parent, pred_data_dir, output_dir, matched=matched)