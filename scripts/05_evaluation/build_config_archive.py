"""Build a git-ignored configs/ archive of the tracking/seg/flow configs referenced
by merge_ablation.toml, solver_ablation.toml and solver_addition.toml.

The archive mirrors experiments/ and uses human-readable, condition-based names so
any condition can be re-run on future code. Run UIDs in the TOMLs are unchanged;
each config file is a verbatim copy of the run's saved config.

Naming (tracking): one file per unique referenced tracking_uid, named by its
condition (first reference wins, in merge -> solver_ablation -> solver_addition
order). If two different uids would collide on the same condition name within a
dataset (the deferred nc281/nc281_sparse merge-vs-solver cases), each is suffixed
with its experiment group (_merge / _solver).

Also prints a (toml, experiment, dataset, condition) -> archive path mapping, used
to add `config` pointers to the TOMLs (build_config_archive.py --print-map).

Usage: python scripts/05_evaluation/build_config_archive.py [--print-map]
"""
import argparse
import os
import shutil
import sys
from collections import OrderedDict

import toml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(REPO, "experiments", "evaluation")
SEG = os.path.join(REPO, "experiments", "segmentation")
FLOW = os.path.join(REPO, "experiments", "opticalflow")
CONFIGS = os.path.join(REPO, "configs")

TOMLS = OrderedDict([
    ("merge", "scripts/05_evaluation/merge_ablation.toml"),
    ("solver_ablation", "scripts/05_evaluation/solver_ablation.toml"),
    ("solver_addition", "scripts/05_evaluation/solver_addition.toml"),
])
GROUP = {"merge": "merge", "solver_ablation": "solver", "solver_addition": "solver"}


def parse_refs():
    """Return list of dicts: {toml, exp, ds, cond, uid, seg_uid}."""
    refs = []
    for ttype, rel in TOMLS.items():
        cfg = toml.load(os.path.join(REPO, rel))
        for dskey, d in cfg.items():
            if not isinstance(d, dict) or "conditions" not in d:
                continue
            exp, ds = d["experiment"], d["dataset_dir"]
            for cond, cc in d["conditions"].items():
                uid = cc.get("tracking_uid", "")
                if not uid or uid == "<fill in>":
                    continue
                refs.append({"toml": ttype, "exp": exp, "ds": ds, "cond": cond,
                             "uid": uid, "seg_uid": cc.get("seg_uid")})
    return refs


def name_tracking(refs):
    """uid -> archive basename, per (exp,ds). Returns {(exp,ds,uid): basename}."""
    # base name = first condition that referenced the uid (TOMLS order preserved)
    base = {}            # (exp,ds,uid) -> condition name
    groups = {}          # (exp,ds,uid) -> set of experiment groups
    for r in refs:
        key = (r["exp"], r["ds"], r["uid"])
        base.setdefault(key, r["cond"])
        groups.setdefault(key, set()).add(GROUP[r["toml"]])
    # detect name collisions within a dataset (same base name, different uids)
    name_to_keys = {}
    for key, nm in base.items():
        name_to_keys.setdefault((key[0], key[1], nm), []).append(key)
    final = {}
    for (exp, ds, nm), keys in name_to_keys.items():
        if len(keys) == 1:
            final[keys[0]] = nm
        else:
            for key in keys:                       # collision -> suffix by group
                g = "merge" if groups[key] == {"merge"} else "solver"
                final[key] = f"{nm}_{g}"
    return final


def copy_verbatim(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    assert open(src, "rb").read() == open(dst, "rb").read(), f"copy mismatch {dst}"


def build():
    refs = parse_refs()
    track_names = name_tracking(refs)
    cond_to_path = {}  # (toml,exp,ds,cond) -> config relpath

    # ---- tracking ----
    seen = set()
    for r in refs:
        key = (r["exp"], r["ds"], r["uid"])
        base = track_names[key]
        rel = f"configs/tracking/{r['exp']}/{r['ds']}/{base}_config.toml"
        cond_to_path[(r["toml"], r["exp"], r["ds"], r["cond"])] = rel
        if key in seen:
            continue
        seen.add(key)
        src = os.path.join(EVAL, r["exp"], r["ds"], r["uid"], "tracking_config.toml")
        copy_verbatim(src, os.path.join(REPO, rel))

    # ---- segmentation (roles from the merge conditions) ----
    seg_role = {}  # (exp,ds,seg_uid) -> role
    for r in refs:
        if r["toml"] == "merge" and r["seg_uid"]:
            role = {"baseline": "baseline", "no_cohesion": "baseline",
                    "no_affinities": "no_affinities", "no_merges": "no_merges"}.get(r["cond"])
            if role:
                seg_role[(r["exp"], r["ds"], r["seg_uid"])] = role
    for (exp, ds, seg_uid), role in seg_role.items():
        src = os.path.join(SEG, exp, ds, seg_uid, "config.toml")
        copy_verbatim(src, os.path.join(CONFIGS, "segmentation", exp, ds, f"{role}_config.toml"))

    # ---- optical flow (one per dataset; 2d + 3d) ----
    flows = {}  # (exp,ds) -> flow_uid (read from each dataset's tracking config)
    for r in refs:
        tc = toml.load(os.path.join(EVAL, r["exp"], r["ds"], r["uid"], "tracking_config.toml"))
        if tc.get("flow_result"):
            flows[(r["exp"], r["ds"])] = tc["flow_result"]
    for (exp, ds), flow_uid in flows.items():
        for sub in ("opticalflow_2d", "opticalflow_3d"):
            src = os.path.join(FLOW, exp, ds, sub, flow_uid, "config.toml")
            if os.path.isfile(src):
                copy_verbatim(src, os.path.join(CONFIGS, "opticalflow", exp, ds, sub, "flow_config.toml"))

    return cond_to_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-map", action="store_true", help="print condition->config mapping as JSON")
    args = ap.parse_args()
    cond_to_path = build()
    if args.print_map:
        import json
        print(json.dumps({f"{t}|{e}|{d}|{c}": p for (t, e, d, c), p in cond_to_path.items()}, indent=2))
    else:
        n = len({p for p in cond_to_path.values()})
        print(f"Built configs/ archive: {len(cond_to_path)} condition refs -> {n} tracking config files")
        print("Run with --print-map to get the condition->config mapping.")
