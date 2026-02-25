#!/usr/bin/env python3
"""Dump event information from an HDF5 output file.

Usage:
    python dump_event.py <file.h5>              # Show file summary
    python dump_event.py <file.h5> 0            # Dump event 0
    python dump_event.py <file.h5> 0 5 10       # Dump events 0, 5, 10
    python dump_event.py <file.h5> --all        # Dump all events (careful with large files)
"""
import sys
import argparse
import numpy as np
import h5py


def print_file_summary(f):
    """Print overall file structure and metadata."""
    print("=" * 70)
    print(f"FILE SUMMARY")
    print("=" * 70)

    # Metadata
    if f.attrs:
        print("\nMetadata:")
        for k, v in sorted(f.attrs.items()):
            print(f"  {k}: {v}")

    # Walk all groups/datasets
    print("\nDatasets:")
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  /{name:40s}  shape={str(obj.shape):20s}  dtype={obj.dtype}")
    f.visititems(visit)

    n_events = f.attrs.get("n_events", None)
    if n_events is None:
        # Try to infer from dataset shapes
        for name in ["event_features", "INPUTS/Source/pt", "INPUTS/Jets/pt", "source/pt"]:
            if name in f:
                n_events = f[name].shape[0]
                break
    print(f"\nTotal events: {n_events}")
    return n_events


def dump_event(f, idx):
    """Dump all information for a single event."""
    print(f"\n{'=' * 70}")
    print(f"EVENT {idx}")
    print(f"{'=' * 70}")

    # --- INPUTS/Source (SPANet format) ---
    for group_name in ["INPUTS/Source", "INPUTS/Jets"]:
        if group_name in f:
            grp = f[group_name]
            mask = grp["MASK"][idx] if "MASK" in grp else None
            n_obj = int(mask.sum()) if mask is not None else None

            print(f"\n  {group_name}/ ({n_obj} active objects):")
            print(f"    {'idx':>4s}  {'pt':>10s}  {'eta':>10s}  {'phi':>10s}  {'mass':>10s}  {'btag':>5s}  {'MASK':>5s}")
            print(f"    {'---':>4s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>5s}  {'---':>5s}")

            n_slots = grp["pt"][idx].shape[0] if "pt" in grp else 0
            for j in range(n_slots):
                active = mask[j] if mask is not None else True
                if not active:
                    continue
                pt = grp["pt"][idx][j] if "pt" in grp else 0
                eta = grp["eta"][idx][j] if "eta" in grp else 0
                phi = grp["phi"][idx][j] if "phi" in grp else 0
                mass = grp["mass"][idx][j] if "mass" in grp else 0
                btag = grp["btag"][idx][j] if "btag" in grp else 0
                print(f"    {j:4d}  {pt:10.2f}  {eta:10.4f}  {phi:10.4f}  {mass:10.4f}  {btag:5.0f}  {bool(active):>5}")
            break

    # --- TARGETS (SPANet assignment targets) ---
    if "TARGETS" in f:
        print(f"\n  TARGETS/:")
        targets_grp = f["TARGETS"]
        for gluino_name in sorted(targets_grp.keys()):
            g = targets_grp[gluino_name]
            jets = []
            for jet_name in sorted(g.keys()):
                jets.append(f"{jet_name}={g[jet_name][idx]}")
            print(f"    {gluino_name}: {', '.join(jets)}")

    # --- source/ (PasswdABC format) ---
    if "source" in f:
        grp = f["source"]
        mask = grp["mask"][idx] if "mask" in grp else None
        n_obj = int(mask.sum()) if mask is not None else None

        print(f"\n  source/ (PasswdABC, {n_obj} active):")
        print(f"    {'idx':>4s}  {'e':>10s}  {'pt':>10s}  {'eta':>10s}  {'phi':>10s}  {'mask':>5s}")
        print(f"    {'---':>4s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>5s}")

        n_slots = grp["pt"][idx].shape[0] if "pt" in grp else 0
        for j in range(n_slots):
            active = mask[j] if mask is not None else True
            if not active:
                continue
            e = grp["e"][idx][j] if "e" in grp else 0
            pt = grp["pt"][idx][j] if "pt" in grp else 0
            eta = grp["eta"][idx][j] if "eta" in grp else 0
            phi = grp["phi"][idx][j] if "phi" in grp else 0
            print(f"    {j:4d}  {e:10.2f}  {pt:10.2f}  {eta:10.4f}  {phi:10.4f}  {bool(active):>5}")

    # --- EventVars ---
    if "EventVars" in f:
        print(f"\n  EventVars/:")
        for k in sorted(f["EventVars"].keys()):
            print(f"    {k}: {f['EventVars'][k][idx]}")

    # --- Legacy event_features ---
    if "event_features" in f:
        ef = f["event_features"][idx]
        feat_names = list(f.attrs.get("event_features", []))
        print(f"\n  event_features (legacy):")
        if feat_names and len(feat_names) == len(ef):
            for name, val in zip(feat_names, ef):
                print(f"    {name}: {val:.4f}")
        else:
            print(f"    {ef}")

    # --- Legacy jet_features ---
    if "jet_features" in f and "jet_mask" in f:
        jf = f["jet_features"][idx]
        jm = f["jet_mask"][idx]
        feat_names = list(f.attrs.get("jet_features", []))
        n_jets = int(jm.sum())
        print(f"\n  jet_features (legacy, {n_jets} jets):")
        if feat_names:
            print(f"    {'idx':>4s}  " + "  ".join(f"{n:>12s}" for n in feat_names))
        for j in range(n_jets):
            vals = "  ".join(f"{v:12.4f}" for v in jf[j])
            print(f"    {j:4d}  {vals}")


def main():
    parser = argparse.ArgumentParser(description="Dump HDF5 event data")
    parser.add_argument("file", help="HDF5 file path")
    parser.add_argument("events", nargs="*", help="Event indices to dump (default: summary only)")
    parser.add_argument("--all", action="store_true", help="Dump all events")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        n_events = print_file_summary(f)

        if args.all:
            indices = range(n_events)
        elif args.events:
            indices = [int(e) for e in args.events]
        else:
            indices = []

        for idx in indices:
            if idx < 0 or idx >= n_events:
                print(f"\nWarning: event {idx} out of range [0, {n_events-1}], skipping")
                continue
            dump_event(f, idx)

    print()


if __name__ == "__main__":
    main()
