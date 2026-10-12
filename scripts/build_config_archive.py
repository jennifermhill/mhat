"""Build a git-ignored configs/ archive of the tracking/seg/flow configs referenced
by merge_ablation.toml, solver_ablation.toml and solver_addition.toml.

The archive is keyed by experiment/dataset (``configs/tracking/<exp>/<ds>/...``,
named from each spec table's ``output_dir``) and uses human-readable,
condition-based names so any condition can be re-run on future code. Each config
file is a verbatim copy of a run's saved ``config.toml``: the condition's run
(``<output_dir>/<run>``, see scripts/07_plotting/plot_spec.py), and the
segmentation and flow runs it read (its ``seg_result`` / ``flow_result``).

Naming (tracking): one file per unique referenced tracking_uid, named by its
condition (first reference wins, in merge -> solver_ablation -> solver_addition
order). If two different uids would collide on the same condition name within a
dataset (the deferred nc281/nc281_sparse merge-vs-solver cases), each is suffixed
with its experiment group (_merge / _solver).

Also prints a (toml, experiment, dataset, condition) -> archive path mapping, used
to add `config` pointers to the TOMLs (build_config_archive.py --print-map).

Usage: python scripts/build_config_archive.py [--print-map]
           [--specs-dir DIR] [--out-root DIR] [--only TABLE ...]

``--specs-dir`` / ``--out-root`` let it run on converted specs and write somewhere
other than configs/ (configs/ is the tracked archive: only with Jennifer's go-ahead).
"""
import argparse
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import toml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOMLS = OrderedDict([
    ("merge", "merge_ablation.toml"),
    ("solver_ablation", "solver_ablation.toml"),
    ("solver_addition", "solver_addition.toml"),
])
GROUP = {"merge": "merge", "solver_ablation": "solver", "solver_addition": "solver"}
PLACEHOLDER = "<fill in>"


def config_source(output_dir, run, eval_name):
    """The run's saved config: its own ``config.toml``, or the copy its eval kept.

    A bar can rest on an *orphan eval* -- the tracking run was deleted at some
    point and the migration put its evaluation under
    ``tracking/_orphan_evals/<uid>/``. There is no run directory to read, but
    every eval saves the tracking config beside its metrics, and that copy is
    what the archive was built from before. 10 NC281-Fl2mSiH2B 03_nuclei solver
    bars are in this state.
    """
    own = Path(output_dir) / run / "config.toml"
    if own.is_file():
        return own
    saved = Path(output_dir) / run / "evaluation" / (eval_name or "") / "tracking_config.toml"
    if eval_name and saved.is_file():
        return saved
    raise SystemExit(
        f"no saved config for run {run}: neither {own} nor {saved}"
    )


def parse_refs(specs_dir, only=None):
    """Return list of dicts: {toml, exp, ds, cond, uid, output_dir, run, seg_uid}.

    ``uid`` is the condition's run (a path relative to ``output_dir``); ``exp`` /
    ``ds`` are the last two parts of ``output_dir``; ``seg_uid`` comes from the
    run's own config (its ``seg_result``), and ``config_src`` is where that config
    was read from.
    """
    refs = []
    for ttype, name in TOMLS.items():
        cfg = toml.load(os.path.join(specs_dir, name))
        for table_name, d in cfg.items():
            if not isinstance(d, dict) or "conditions" not in d:
                continue
            if only and table_name not in only:
                continue
            if "output_dir" not in d:
                raise SystemExit(
                    f"{name} [{table_name}] has no output_dir: convert the spec first "
                    "(scripts/convert_figure_specs.py)"
                )
            output_dir = Path(d["output_dir"])
            exp, ds = output_dir.parts[-2], output_dir.parts[-1]
            for cond, cc in d["conditions"].items():
                run = cc.get("run", "")
                if not run or run == PLACEHOLDER:
                    continue
                config_src = config_source(output_dir, run, cc.get("eval"))
                run_config = toml.load(config_src)
                refs.append({"toml": ttype, "exp": exp, "ds": ds, "cond": cond,
                             "uid": run, "output_dir": output_dir,
                             "config_src": config_src,
                             "run_config": run_config,
                             "seg_uid": run_config.get("seg_result")})
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
    for (_exp, _ds, nm), keys in name_to_keys.items():
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


def build(specs_dir, out_root, only=None):
    refs = parse_refs(specs_dir, only)
    track_names = name_tracking(refs)
    cond_to_path = {}  # (toml,exp,ds,cond) -> config relpath

    # ---- tracking ----
    seen = set()
    for r in refs:
        key = (r["exp"], r["ds"], r["uid"])
        base = track_names[key]
        rel = f"tracking/{r['exp']}/{r['ds']}/{base}_config.toml"
        cond_to_path[(r["toml"], r["exp"], r["ds"], r["cond"])] = f"configs/{rel}"
        if key in seen:
            continue
        seen.add(key)
        copy_verbatim(r["config_src"], os.path.join(out_root, rel))

    # ---- segmentation (roles from the merge conditions) ----
    seg_role = {}  # (exp,ds,seg_uid) -> (role, output_dir)
    for r in refs:
        if r["toml"] == "merge" and r["seg_uid"]:
            role = {"baseline": "baseline", "no_cohesion": "baseline",
                    "no_affinities": "no_affinities", "no_merges": "no_merges"}.get(r["cond"])
            if role:
                seg_role[(r["exp"], r["ds"], r["seg_uid"])] = (role, r["output_dir"])
    for (exp, ds, seg_uid), (role, output_dir) in seg_role.items():
        src = output_dir / "segmentation" / seg_uid / "config.toml"
        copy_verbatim(src, os.path.join(out_root, "segmentation", exp, ds, f"{role}_config.toml"))

    # ---- optical flow (one per dataset; 2d + 3d) ----
    flows = {}  # (exp,ds) -> (flow_uid, output_dir), from each run's own config
    for r in refs:
        if r["run_config"].get("flow_result"):
            flows[(r["exp"], r["ds"])] = (r["run_config"]["flow_result"], r["output_dir"])
    # The archive keeps its opticalflow_{2d,3d} names; the runs live in
    # <output_dir>/opticalflow/{2d,3d}/.
    for (exp, ds), (flow_uid, output_dir) in flows.items():
        for kind in ("2d", "3d"):
            src = output_dir / "opticalflow" / kind / flow_uid / "config.toml"
            if src.is_file():
                copy_verbatim(src, os.path.join(out_root, "opticalflow", exp, ds,
                                                f"opticalflow_{kind}", "flow_config.toml"))

    return cond_to_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-map", action="store_true", help="print condition->config mapping as JSON")
    ap.add_argument("--specs-dir", default=os.path.join(REPO, "configs", "evaluation"),
                    help="where the (converted) figure specs are")
    ap.add_argument("--out-root", default=os.path.join(REPO, "configs"),
                    help="archive root to write (default: the tracked configs/)")
    ap.add_argument("--only", nargs="+", default=None, help="only these spec tables")
    args = ap.parse_args()
    cond_to_path = build(args.specs_dir, args.out_root, args.only)
    if args.print_map:
        import json
        print(json.dumps({f"{t}|{e}|{d}|{c}": p for (t, e, d, c), p in cond_to_path.items()}, indent=2))
    else:
        n = len({p for p in cond_to_path.values()})
        print(f"Built configs/ archive: {len(cond_to_path)} condition refs -> {n} tracking config files")
        print("Run with --print-map to get the condition->config mapping.")
