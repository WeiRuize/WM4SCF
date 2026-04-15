"""
scripts/identify_backdoor_demos.py
==================================
One-shot probe that scans Poisoned_Dataset/ and produces a JSON mapping

    {
      "<poisoned_hdf5_path>": {
        "clean":    ["demo_0", "demo_3", ...],
        "backdoor": ["demo_7", "demo_12", ...]
      },
      ...
    }

Identification strategy (two-stage fallback):

  Stage 1  original_name set difference.
           For each poisoned file we open the matching clean file in
           `no_noops_datasets/` and read the set of demo keys
           {demo_0, ..., demo_{N-1}}.  Every demo in the poisoned file
           whose `attrs["original_name"]` is NOT in that set must come
           from the BadLIBERO Poison source — it is a backdoor demo.

  Stage 2  action-sequence hashing (only if Stage 1 returns 0 backdoors,
           which means BadLIBERO reused the demo_{i} naming).
           For every demo in the poisoned file we MD5 its `actions`
           array and check whether an identical hash exists among the
           clean file's demos.  Demos whose action sequence has NO
           counterpart in the clean file are backdoor demos.

Either stage, augmented by the counts in inject_log.txt, gives a
definitive label — we cross-check the identified backdoor count against
inject_log to catch regressions.

Run on the server (where the HDF5 files are not truncated):

    python scripts/identify_backdoor_demos.py \
        --clean_root  /path/to/no_noops_datasets \
        --poison_root /path/to/Poisoned_Dataset \
        --inject_log  /path/to/Poisoned_Dataset/inject_log.txt \
        --out         data/backdoor_index.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import re
import sys

import h5py
import numpy as np


INJECT_LINE = re.compile(
    r"Injecting file (?P<suite>[^/]+)/(?P<file>[^:]+): clean (?P<clean>\d+), backdoor (?P<bd>\d+)"
)


def parse_inject_log(path: pathlib.Path) -> dict[tuple[str, str], tuple[int, int]]:
    """Return {(suite, file): (clean_count, backdoor_count)}."""
    out: dict[tuple[str, str], tuple[int, int]] = {}
    with path.open() as f:
        for line in f:
            m = INJECT_LINE.search(line)
            if m:
                out[(m.group("suite"), m.group("file"))] = (
                    int(m.group("clean")), int(m.group("bd")),
                )
    return out


def md5_array(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def classify_one_file(
    clean_fp: pathlib.Path,
    poison_fp: pathlib.Path,
) -> tuple[list[str], list[str], str]:
    """
    Returns (clean_demo_keys, backdoor_demo_keys, stage_used).
    `clean_demo_keys` and `backdoor_demo_keys` are keys inside the
    POISONED file, not the clean file.
    """
    # Stage 1: original_name set difference
    with h5py.File(clean_fp, "r") as cf:
        clean_original_names = set(cf["data"].keys())
        clean_action_hashes = {
            md5_array(cf[f"data/{k}/actions"][()]): k for k in cf["data"].keys()
        }

    clean_keys: list[str] = []
    backdoor_keys: list[str] = []
    stage_used = "stage1_original_name"

    with h5py.File(poison_fp, "r") as pf:
        demos = sorted(pf["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for d in demos:
            grp = pf[f"data/{d}"]
            orig = grp.attrs.get("original_name", None)
            if orig is not None and orig in clean_original_names:
                clean_keys.append(d)
            else:
                backdoor_keys.append(d)

        if not backdoor_keys:
            # Stage 2 fallback: BadLIBERO may have reused demo_{i} names.
            stage_used = "stage2_action_hash"
            clean_keys, backdoor_keys = [], []
            for d in demos:
                h = md5_array(pf[f"data/{d}/actions"][()])
                if h in clean_action_hashes:
                    clean_keys.append(d)
                else:
                    backdoor_keys.append(d)

    return clean_keys, backdoor_keys, stage_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--poison_root", required=True)
    parser.add_argument("--inject_log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    clean_root = pathlib.Path(args.clean_root)
    poison_root = pathlib.Path(args.poison_root)

    expected = parse_inject_log(pathlib.Path(args.inject_log))
    print(f"[identify] inject_log covers {len(expected)} files")

    index: dict[str, dict] = {}
    mismatches: list[str] = []

    for poison_fp in sorted(poison_root.rglob("*.hdf5")):
        rel = poison_fp.relative_to(poison_root)
        suite, fname = rel.parts[0], rel.parts[-1]
        clean_fp = clean_root / suite / fname
        if not clean_fp.exists():
            print(f"  skip {rel} — no matching clean file")
            continue

        clean_keys, backdoor_keys, stage = classify_one_file(clean_fp, poison_fp)
        n_clean, n_bd = len(clean_keys), len(backdoor_keys)

        tag = f"{suite}/{fname}"
        key = (suite, fname)
        expected_pair = expected.get(key)
        if expected_pair is None:
            status = "no_log"
        elif expected_pair == (n_clean, n_bd):
            status = "ok"
        else:
            status = f"mismatch (log={expected_pair}, found=({n_clean},{n_bd}))"
            mismatches.append(tag)

        print(f"  [{stage}] {tag}: clean={n_clean}, backdoor={n_bd}  {status}")
        index[str(poison_fp)] = {
            "clean": clean_keys,
            "backdoor": backdoor_keys,
            "stage": stage,
        }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(index, f, indent=2)

    total_bd = sum(len(v["backdoor"]) for v in index.values())
    print(f"\n[identify] wrote {out_path}")
    print(f"[identify] total backdoor demos identified: {total_bd}")
    print(f"[identify] log expected total: "
          f"{sum(bd for _, bd in expected.values())}")
    if mismatches:
        print(f"[identify] WARNING: {len(mismatches)} files failed cross-check")
        sys.exit(1)


if __name__ == "__main__":
    main()
