#!/usr/bin/env bash
# 01_preprocess.sh — sanity-check datasets before training anything.
# Prints HDF5 layout for clean and poisoned splits so downstream loaders
# can assume the expected key names.
set -euo pipefail

python - <<'PY'
import h5py, pathlib
for tag, root in (("clean", "no_noops_datasets"),
                  ("poison", "Poisoned_Dataset")):
    root = pathlib.Path(root)
    if not root.exists():
        print(f"[{tag}] missing: {root}")
        continue
    files = sorted(root.rglob("*.hdf5"))
    print(f"[{tag}] {len(files)} files under {root}")
    if not files:
        continue
    with h5py.File(files[0], "r") as f:
        def walk(name, obj):
            shape = getattr(obj, "shape", "")
            print(f"    {name} {shape}")
        f.visititems(walk)
    print()
PY
