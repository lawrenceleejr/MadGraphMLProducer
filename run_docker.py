#!/usr/bin/env python3
"""
Event generation pipeline - runs inside Docker container.

This script is the entrypoint for the all-in-one Docker image.
It runs MadGraph5, Pythia8 (via MadGraph), and converts to HDF5.
"""

import argparse
import shutil
import subprocess
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate particle physics events for ML training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="/app/configs/examples/gluino_rpv_1tev.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--events", "-n",
        type=int,
        default=10000,
        help="Number of events to generate",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/app/output/events.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=12345,
        help="Random seed",
    )
    parser.add_argument(
        "--cores", "-j",
        type=int,
        default=4,
        help="Number of CPU cores for MadGraph",
    )
    parser.add_argument(
        "--shower",
        type=str,
        choices=["pythia8", "off"],
        default="pythia8",
        help="Parton shower program",
    )
    parser.add_argument(
        "--keep-lhe",
        action="store_true",
        help="Keep intermediate LHE file in output directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MadGraphMLProducer - Containerized Event Generation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Events: {args.events}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Cores: {args.cores}")
    print(f"Shower: {args.shower}")
    print("=" * 60)

    # Set up paths
    work_dir = Path("/app/work")
    work_dir.mkdir(exist_ok=True)

    # Clean work directory
    for item in work_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Step 1: Generate cards
    print("\n[1/3] Generating MadGraph configuration...")
    cards_dir = generate_cards(config_path, work_dir, args.events, args.seed)

    # Step 2: Run MadGraph + Pythia8
    print("\n[2/3] Running MadGraph5 + Pythia8...")
    lhe_file, hepmc_file = run_madgraph_pythia(
        cards_dir, work_dir, args.cores, args.shower, args.verbose
    )

    # Step 3: Convert to HDF5
    print("\n[3/3] Converting to HDF5...")
    output_path = Path(args.output)
    convert_to_hdf5(hepmc_file or lhe_file, output_path, config_path)

    # Optionally copy LHE file
    if args.keep_lhe and lhe_file and lhe_file.exists():
        lhe_dest = output_path.parent / "events.lhe.gz"
        shutil.copy(lhe_file, lhe_dest)
        print(f"LHE file saved to: {lhe_dest}")

    print("\n" + "=" * 60)
    print(f"SUCCESS!")
    print(f"Output: {output_path}")
    print("=" * 60)


def generate_cards(config_path: Path, work_dir: Path, num_events: int, seed: int) -> Path:
    """Generate MadGraph and Pythia8 cards."""
    import yaml
    sys.path.insert(0, "/app/src")
    from madgraph_ml_producer.config import PipelineConfig
    from madgraph_ml_producer.card_generator import CardGenerator

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Override settings
    config_dict["generation"]["num_events"] = num_events
    config_dict["generation"]["random_seed"] = seed

    config = PipelineConfig(**config_dict)

    cards_dir = work_dir / "cards"
    generator = CardGenerator(
        config,
        template_dir=Path("/app/cards/templates"),
        output_dir=cards_dir,
    )
    output_dir = generator.generate_all_cards()

    print(f"  Generated cards in: {output_dir}")
    return output_dir


def run_madgraph_pythia(cards_dir: Path, work_dir: Path, cores: int,
                        shower: str, verbose: bool) -> tuple:
    """Run MadGraph5 with optional Pythia8 shower."""
    mg5_output = work_dir / "mg5_run"
    mg5_output.mkdir(exist_ok=True)

    # Read process name from proc_card
    proc_card = cards_dir / "proc_card.dat"
    process_name = "signal"
    with open(proc_card) as f:
        for line in f:
            if line.strip().startswith("output"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    process_name = parts[1]
                break

    # Create MadGraph launch script
    mg5_script = work_dir / "mg5_run.txt"

    # Build the script content
    script_lines = [
        f"import {cards_dir}/proc_card.dat",
        f"launch {process_name}",
    ]

    if shower == "pythia8":
        script_lines.append("  shower=Pythia8")
    else:
        script_lines.append("  shower=OFF")

    script_lines.extend([
        "  done",
        f"  {cards_dir}/param_card.dat",
        f"  {cards_dir}/run_card.dat",
    ])

    if shower == "pythia8":
        script_lines.append(f"  {cards_dir}/pythia8_card.dat")

    script_lines.append("  done")

    with open(mg5_script, "w") as f:
        f.write("\n".join(script_lines))

    # Set environment for multicore
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cores)

    # Run MadGraph
    print(f"  Running mg5_aMC with {cores} cores...")
    cmd = ["mg5_aMC", str(mg5_script)]

    if verbose:
        result = subprocess.run(cmd, cwd=mg5_output, env=env)
    else:
        result = subprocess.run(
            cmd, cwd=mg5_output, env=env,
            capture_output=True, text=True
        )

    if result.returncode != 0:
        print("MadGraph5 failed!")
        if not verbose and hasattr(result, 'stderr'):
            print(result.stderr[-3000:] if result.stderr else "No stderr")
        sys.exit(1)

    # Find output files
    lhe_file = None
    hepmc_file = None

    # Look for LHE file
    lhe_candidates = list(mg5_output.glob("**/unweighted_events.lhe.gz"))
    if not lhe_candidates:
        lhe_candidates = list(mg5_output.glob("**/events.lhe.gz"))
    if not lhe_candidates:
        lhe_candidates = list(mg5_output.glob("**/*.lhe.gz"))
    if not lhe_candidates:
        lhe_candidates = list(mg5_output.glob("**/*.lhe"))

    if lhe_candidates:
        lhe_file = lhe_candidates[0]
        print(f"  LHE file: {lhe_file}")

    # Look for HepMC file (from Pythia8 shower)
    hepmc_candidates = list(mg5_output.glob("**/*.hepmc"))
    if not hepmc_candidates:
        hepmc_candidates = list(mg5_output.glob("**/*.hepmc.gz"))

    if hepmc_candidates:
        hepmc_file = hepmc_candidates[0]
        print(f"  HepMC file: {hepmc_file}")

    if not lhe_file and not hepmc_file:
        print("Error: No output files found!")
        print(f"Contents of {mg5_output}:")
        for f in mg5_output.rglob("*"):
            if f.is_file():
                print(f"  {f}")
        sys.exit(1)

    return lhe_file, hepmc_file


def convert_to_hdf5(input_file: Path, output_path: Path, config_path: Path):
    """Convert LHE or HepMC to HDF5."""
    import yaml
    sys.path.insert(0, "/app/src")
    from madgraph_ml_producer.config import PipelineConfig
    from madgraph_ml_producer.hdf5_writer import HDF5Writer
    from madgraph_ml_producer.truth_extractor import TruthExtractor

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = PipelineConfig(**config_dict)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check input file type
    input_suffix = input_file.suffix.lower()
    if input_suffix == ".gz":
        # Check the actual format
        stem_suffix = input_file.stem.split(".")[-1].lower()
        is_hepmc = stem_suffix == "hepmc"
    else:
        is_hepmc = input_suffix == ".hepmc"

    if is_hepmc:
        # Direct HepMC processing
        writer = HDF5Writer(config)
        extractor = TruthExtractor(config)
        writer.process_file(input_file, output_path, extractor)
    else:
        # LHE file - need to convert via pyhepmc or process directly
        print("  Processing LHE file (no parton shower)...")
        process_lhe_to_hdf5(input_file, output_path, config)

    print(f"  Wrote: {output_path}")


def process_lhe_to_hdf5(lhe_file: Path, output_path: Path, config):
    """Process LHE file directly to HDF5 (parton-level, no shower)."""
    import gzip
    import numpy as np
    import h5py
    from tqdm import tqdm

    try:
        import pylhe
    except ImportError:
        print("Error: pylhe not installed")
        sys.exit(1)

    # Handle compressed files
    if str(lhe_file).endswith(".gz"):
        # pylhe can handle gzipped files directly
        pass

    print(f"  Reading LHE file: {lhe_file}")

    # Count events first
    events_list = list(pylhe.read_lhe_with_attributes(str(lhe_file)))
    n_events = len(events_list)
    print(f"  Found {n_events} events")

    # Set up HDF5 output
    max_jets = config.output.max_jets_per_event
    max_particles = config.output.max_particles_per_jet

    PARTICLE_FEATURES = ["pt_rel", "eta_rel", "phi_rel", "energy", "pdg_id"]
    JET_FEATURES = ["pt", "eta", "phi", "mass", "n_constituents", "parent_pdg", "is_signal"]
    EVENT_FEATURES = ["n_jets", "met_x", "met_y", "met_pt", "ht", "n_signal", "weight"]

    with h5py.File(output_path, "w") as f:
        # Create datasets
        compression = config.output.compression

        f.create_dataset(
            "particle_features",
            shape=(n_events, max_jets, max_particles, len(PARTICLE_FEATURES)),
            dtype=np.float32,
            compression=compression,
        )
        f.create_dataset(
            "jet_features",
            shape=(n_events, max_jets, len(JET_FEATURES)),
            dtype=np.float32,
            compression=compression,
        )
        f.create_dataset(
            "event_features",
            shape=(n_events, len(EVENT_FEATURES)),
            dtype=np.float32,
            compression=compression,
        )
        f.create_dataset(
            "jet_mask",
            shape=(n_events, max_jets),
            dtype=bool,
            compression=compression,
        )
        f.create_dataset(
            "particle_mask",
            shape=(n_events, max_jets, max_particles),
            dtype=bool,
            compression=compression,
        )

        # Process events
        for i, event in enumerate(tqdm(events_list, desc="Processing events")):
            # Get final state particles (status == 1)
            final_particles = [p for p in event.particles if p.status == 1]

            # Simple "jet" = each final state parton (at LHE level, no clustering)
            # For proper jets, we'd need Pythia8 shower first

            jets_data = np.zeros((max_jets, len(JET_FEATURES)), dtype=np.float32)
            particles_data = np.zeros((max_jets, max_particles, len(PARTICLE_FEATURES)), dtype=np.float32)
            jet_mask = np.zeros(max_jets, dtype=bool)
            particle_mask = np.zeros((max_jets, max_particles), dtype=bool)

            met_x, met_y = 0.0, 0.0
            ht = 0.0
            n_signal = 0

            # Check for signal particles (gluinos, etc.)
            signal_pdg_ids = {1000021, -1000021}  # gluino
            for p in event.particles:
                if abs(p.id) in signal_pdg_ids:
                    n_signal += 1

            # Process final state particles as "jets" (parton-level)
            jet_idx = 0
            for p in final_particles:
                # Skip neutrinos for visible particles
                if abs(p.id) in {12, 14, 16}:
                    met_x += p.px
                    met_y += p.py
                    continue

                if jet_idx >= max_jets:
                    break

                pt = np.sqrt(p.px**2 + p.py**2)
                pz = p.pz
                p_tot = np.sqrt(p.px**2 + p.py**2 + p.pz**2)
                eta = np.arctanh(np.clip(pz / (p_tot + 1e-10), -0.9999, 0.9999))
                phi = np.arctan2(p.py, p.px)
                mass = np.sqrt(max(0, p.e**2 - p_tot**2))

                # Apply cuts
                if pt < config.jets.pt_min or abs(eta) > config.jets.eta_max:
                    continue

                jets_data[jet_idx] = [pt, eta, phi, mass, 1, 0, 0]
                jet_mask[jet_idx] = True

                # Single particle as constituent
                particles_data[jet_idx, 0] = [1.0, 0.0, 0.0, p.e, p.id]
                particle_mask[jet_idx, 0] = True

                ht += pt
                jet_idx += 1

            n_jets = jet_idx
            met_pt = np.sqrt(met_x**2 + met_y**2)

            event_data = np.array([n_jets, met_x, met_y, met_pt, ht, n_signal, 1.0], dtype=np.float32)

            # Write to HDF5
            f["jet_features"][i] = jets_data
            f["particle_features"][i] = particles_data
            f["event_features"][i] = event_data
            f["jet_mask"][i] = jet_mask
            f["particle_mask"][i] = particle_mask

        # Write metadata
        f.attrs["config_name"] = config.name
        f.attrs["n_events"] = n_events
        f.attrs["particle_features"] = PARTICLE_FEATURES
        f.attrs["jet_features"] = JET_FEATURES
        f.attrs["event_features"] = EVENT_FEATURES
        f.attrs["max_jets"] = max_jets
        f.attrs["max_particles"] = max_particles
        f.attrs["jet_algorithm"] = "none (parton-level)"
        f.attrs["jet_radius"] = 0.0
        f.attrs["process_string"] = config.process.process_string
        f.attrs["model"] = config.process.model


if __name__ == "__main__":
    main()
