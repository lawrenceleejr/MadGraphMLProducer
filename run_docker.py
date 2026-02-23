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
    cards_dir, config = generate_cards(config_path, work_dir, args.events, args.seed, args.shower)

    # Step 2: Run MadGraph + Pythia8
    print("\n[2/3] Running MadGraph5 + Pythia8...")
    lhe_file = run_madgraph_pythia(
        cards_dir, work_dir, config, args.cores, args.shower, args.verbose
    )

    # Step 3: Convert to HDF5
    print("\n[3/3] Converting to HDF5...")
    output_path = Path(args.output)
    convert_to_hdf5(lhe_file, output_path, config)

    # Optionally copy LHE file
    if args.keep_lhe and lhe_file and lhe_file.exists():
        lhe_dest = output_path.parent / "events.lhe.gz"
        shutil.copy(lhe_file, lhe_dest)
        print(f"LHE file saved to: {lhe_dest}")

    print("\n" + "=" * 60)
    print(f"SUCCESS!")
    print(f"Output: {output_path}")
    print("=" * 60)


def generate_cards(config_path: Path, work_dir: Path, num_events: int, seed: int, shower: str = "pythia8") -> tuple:
    """Generate MadGraph and Pythia8 cards."""
    import yaml
    sys.path.insert(0, "/app/src")
    from madgraph_ml_producer.config import PipelineConfig
    from madgraph_ml_producer.card_generator import CardGenerator

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Override settings
    if "generation" not in config_dict:
        config_dict["generation"] = {}
    config_dict["generation"]["num_events"] = num_events
    config_dict["generation"]["random_seed"] = seed

    config = PipelineConfig(**config_dict)

    cards_dir = work_dir / "cards"
    output_dir = cards_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for custom cards
    custom_param = config.custom_param_card
    custom_run = config.custom_run_card

    if custom_param or custom_run:
        print(f"  Using custom cards...")
        # Generate only the cards we need from templates
        generator = CardGenerator(
            config,
            template_dir=Path("/app/cards/templates"),
            output_dir=cards_dir,
            extra_vars={"shower": shower},
        )
        # Always generate proc_card from template
        generator.generate_proc_card(output_dir)
        generator.generate_pythia8_card(output_dir)

        # Copy custom cards or generate from template
        if custom_param:
            print(f"    Using custom param_card: {custom_param}")
            shutil.copy(custom_param, output_dir / "param_card.dat")
        else:
            generator.generate_param_card(output_dir)

        if custom_run:
            print(f"    Using custom run_card: {custom_run}")
            shutil.copy(custom_run, output_dir / "run_card.dat")
        else:
            generator.generate_run_card(output_dir)
    else:
        generator = CardGenerator(
            config,
            template_dir=Path("/app/cards/templates"),
            output_dir=cards_dir,
            extra_vars={"shower": shower},
        )
        output_dir = generator.generate_all_cards()

    print(f"  Generated cards in: {output_dir}")
    return output_dir, config


def patch_param_card(param_card_path: Path, config):
    """Patch specific values in an auto-generated param_card.dat.

    Used for complex UFO models (RPVMSSM etc.) where MadGraph auto-generates
    a full param_card with many blocks. We only update the values specified
    in the config rather than replacing the whole file.
    """
    with open(param_card_path) as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    # Build update maps
    mass_updates = {}
    for particle in config.particles:
        if particle.mass is not None:
            mass_updates[str(particle.pdg_id)] = particle.mass

    # Build decay updates from config.decays
    decay_updates = {}  # {pdg_id: DecayConfig}
    if hasattr(config, 'decays') and config.decays:
        for decay in config.decays:
            decay_updates[str(decay.pdg_id)] = decay

    # If no explicit decays but decay_chain mentions go > u d s, add default gluino decay
    if not decay_updates and hasattr(config.process, 'decay_chain') and config.process.decay_chain:
        decay_chain = config.process.decay_chain.lower()
        if 'go' in decay_chain and ('u' in decay_chain or 'd' in decay_chain or 's' in decay_chain):
            # Create a default gluino decay config
            class DefaultDecay:
                pdg_id = 1000021
                width = 1.0
                class Channel:
                    br = 1.0
                    products = [2, 1, 3]  # u d s
                channels = [Channel()]
            decay_updates['1000021'] = DefaultDecay()

    # RPV coupling updates
    rpv_updates = {}
    if hasattr(config.process, 'rpv_couplings'):
        for coupling in config.process.rpv_couplings:
            rpv_updates[(coupling['block'].upper(), coupling['index'])] = coupling['value']

    current_block = None
    new_lines = []
    skip_decay_channels = False
    current_decay_pdg = None

    for line in lines:
        stripped = line.strip().lower()

        # Detect block header
        if stripped.startswith('block '):
            current_block = stripped.split()[1].upper()
            skip_decay_channels = False
            new_lines.append(line)
            continue

        # Handle DECAY lines
        if stripped.startswith('decay'):
            current_block = None
            skip_decay_channels = False
            parts = stripped.split()
            if len(parts) >= 2:
                pdg = parts[1]
                if pdg in decay_updates:
                    # Replace this decay with our custom one
                    decay_cfg = decay_updates[pdg]
                    new_lines.append(f"DECAY  {pdg}  {decay_cfg.width:.6e}  # custom decay\n")
                    for channel in decay_cfg.channels:
                        products_str = '  '.join(str(p) for p in channel.products)
                        n_products = len(channel.products)
                        new_lines.append(f"   {channel.br:.6e}   {n_products}   {products_str}  # custom channel\n")
                    skip_decay_channels = True
                    current_decay_pdg = pdg
                    continue
            new_lines.append(line)
            continue

        # Skip original decay channels if we're replacing
        if skip_decay_channels:
            # Check if this is still part of the decay (number at start or comment)
            if stripped.startswith('#'):
                continue
            parts = stripped.split()
            if parts and parts[0].replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                continue
            else:
                skip_decay_channels = False

        # Try to patch MASS block
        if current_block == 'MASS' and mass_updates:
            parts = line.split()
            if len(parts) >= 2 and parts[0].lstrip('-').isdigit():
                pdg = parts[0]
                if pdg in mass_updates:
                    comment = ' '.join(parts[2:]) if len(parts) > 2 else ''
                    line = f"  {pdg:>10}  {mass_updates[pdg]:<14.6e}  # {comment}\n"

        # Try to patch RPV coupling blocks
        if current_block in ('RVLAMUDD', 'RVLAMLLE', 'RVLAMLQD') and rpv_updates:
            parts = line.split()
            if len(parts) >= 2:
                idx = parts[0]
                key = (current_block, idx)
                if key in rpv_updates:
                    comment = ' '.join(parts[2:]) if len(parts) > 2 else ''
                    line = f"  {idx:>5}  {rpv_updates[key]:<14.6e}  # {comment}\n"

        new_lines.append(line)

    with open(param_card_path, 'w') as f:
        f.writelines(new_lines)


def run_madgraph_pythia(cards_dir: Path, work_dir: Path, config,
                        cores: int, shower: str, verbose: bool) -> Path:
    """Run MadGraph5 with optional Pythia8 shower."""

    process_name = config.name
    mg5_output = work_dir / "mg5_run"
    mg5_output.mkdir(exist_ok=True)

    # Step 1: Generate the process directory first
    proc_script = work_dir / "mg5_proc.txt"
    proc_card = cards_dir / "proc_card.dat"
    with open(proc_card) as f:
        proc_content = f.read()

    with open(proc_script, "w") as f:
        f.write(proc_content)

    print(f"  Step 1: Generating process code...")
    cmd = ["mg5_aMC", str(proc_script)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cores)

    # Stream output in real-time for visibility
    if verbose:
        print("  --- MadGraph process generation output ---")
        result = subprocess.run(cmd, cwd=mg5_output, env=env)
    else:
        result = subprocess.run(cmd, cwd=mg5_output, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            if result.stdout:
                print("--- MadGraph stdout ---")
                print(result.stdout[-4000:])
            if result.stderr:
                print("--- MadGraph stderr ---")
                print(result.stderr[-2000:])

    if result.returncode != 0:
        print("MadGraph process generation failed!")
        sys.exit(1)

    # Process directory should now exist
    process_dir = mg5_output / process_name
    if not process_dir.exists():
        # Show output to help debug
        if not verbose and hasattr(result, 'stdout'):
            print("--- MadGraph stdout (last 4000 chars) ---")
            print(result.stdout[-4000:] if result.stdout else "(empty)")
            print("--- MadGraph stderr (last 2000 chars) ---")
            print(result.stderr[-2000:] if result.stderr else "(empty)")
        print(f"Error: Process directory not created: {process_dir}")
        sys.exit(1)

    # Step 2: Copy our cards to the Cards directory
    cards_dest = process_dir / "Cards"
    print(f"  Step 2: Copying configuration cards...")

    # run_card and pythia8_card are always replaced with ours
    shutil.copy(cards_dir / "run_card.dat", cards_dest / "run_card.dat")
    if shower == "pythia8" and (cards_dir / "pythia8_card.dat").exists():
        shutil.copy(cards_dir / "pythia8_card.dat", cards_dest / "pythia8_card.dat")

    # For param_card: always patch the auto-generated one rather than replacing it.
    # MadGraph generates a model-correct param_card during process generation.
    # We only update the specific values (masses, couplings) from our config.
    generated_param = cards_dest / "param_card.dat"
    if generated_param.exists():
        print(f"  Patching auto-generated param_card with config values...")
        patch_param_card(generated_param, config)
    else:
        # Fallback: use our template (simple SM models where MG5 may not auto-generate)
        shutil.copy(cards_dir / "param_card.dat", generated_param)

    # Step 3: Run the launch
    print(f"  Step 3: Launching event generation...")
    if not verbose:
        print(f"  This may take several minutes... (use --verbose for live output)")

    # Use generate_events -f directly instead of mg5_aMC interactive launch.
    # This is more reliable as it skips interactive prompts.
    generate_events_bin = process_dir / "bin" / "generate_events"

    if generate_events_bin.exists():
        # Direct invocation with -f (force, no prompts).
        # Shower is configured via parton_shower in run_card.dat.
        cmd = [str(generate_events_bin), "-f"]
        print(f"  Running: {' '.join(cmd)}")
        if verbose:
            print("  --- MadGraph event generation output ---")
            # Use Popen to stream output while also capturing it
            process = subprocess.Popen(
                cmd, cwd=process_dir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            process.wait()
            result_returncode = process.returncode
            result_stdout = ''.join(output_lines)
        else:
            result = subprocess.run(
                cmd, cwd=process_dir, env=env,
                capture_output=True, text=True
            )
            result_returncode = result.returncode
            result_stdout = result.stdout
            result_stderr = getattr(result, 'stderr', '')
    else:
        # Fallback: mg5_aMC launch script
        mg5_script = work_dir / "mg5_launch.txt"
        script_lines = [
            f"launch {process_dir}",
            "shower=Pythia8" if shower == "pythia8" else "shower=OFF",
            "done",  # accept shower
            "done",  # accept cards
        ]
        with open(mg5_script, "w") as f:
            f.write("\n".join(script_lines))
        cmd = ["mg5_aMC", str(mg5_script)]
        print(f"  Running: {' '.join(cmd)}")
        if verbose:
            print("  --- MadGraph event generation output ---")
            process = subprocess.Popen(
                cmd, cwd=mg5_output, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            process.wait()
            result_returncode = process.returncode
            result_stdout = ''.join(output_lines)
        else:
            result = subprocess.run(
                cmd, cwd=mg5_output, env=env,
                capture_output=True, text=True
            )
            result_returncode = result.returncode
            result_stdout = result.stdout
            result_stderr = getattr(result, 'stderr', '')

    print(f"  MadGraph return code: {result_returncode}")

    # Show output summary (only if not verbose, since verbose already showed it)
    if not verbose:
        if result_stdout:
            print("\n  --- MadGraph output (last 3000 chars) ---")
            print(result_stdout[-3000:])
        if result_stderr:
            print("\n  --- MadGraph stderr ---")
            print(result_stderr[-1000:])

    if result_returncode != 0:
        print("MadGraph5 failed with non-zero exit code!")
        sys.exit(1)

    # Find output files - MadGraph creates Events/run_01/unweighted_events.lhe.gz
    # Never pick up intermediate subprocess files from SubProcesses/
    lhe_file = None

    def find_lhe(base: Path):
        """Search for final combined LHE file, excluding SubProcesses directory."""
        # Most specific: standard MadGraph output location
        for pattern in [
            "Events/run_*/unweighted_events.lhe.gz",
            "Events/run_*/unweighted_events.lhe",
            "Events/run_*/*.lhe.gz",
            "Events/run_*/*.lhe",
        ]:
            candidates = [p for p in base.glob(pattern)
                          if "SubProcesses" not in str(p)]
            if candidates:
                return sorted(candidates)[-1]  # latest run
        return None

    lhe_file = find_lhe(process_dir)
    if not lhe_file and (mg5_output / process_name) != process_dir:
        lhe_file = find_lhe(mg5_output / process_name)

    if not lhe_file:
        print("Error: No LHE file generated!")
        print(f"Looking in Events/ under: {process_dir}")
        events_dir = process_dir / "Events"
        if events_dir.exists():
            print(f"\nContents of {events_dir}:")
            for f in events_dir.rglob("*"):
                if f.is_file():
                    print(f"  {f}")
        else:
            print("  Events/ directory does not exist - event generation may have failed")
            # Show top-level process dir contents for debugging
            print(f"\nContents of {process_dir}:")
            for f in process_dir.iterdir():
                print(f"  {f}")
        for item in work_dir.iterdir():
            print(f"  {item}")

        sys.exit(1)

    print(f"  LHE file: {lhe_file}")
    return lhe_file


def convert_to_hdf5(lhe_file: Path, output_path: Path, config):
    """Convert LHE to HDF5 format."""

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Processing LHE file (parton-level)...")
    process_lhe_to_hdf5(lhe_file, output_path, config)
    print(f"  Wrote: {output_path}")


def process_lhe_to_hdf5(lhe_file: Path, output_path: Path, config):
    """Process LHE file to HDF5 with SPANet and PasswdABC compatible formats.

    Creates a unified HDF5 file with multiple format groups:
    - /INPUTS/Jets/: SPANet format (MASK, pt, eta, phi, mass, btag)
    - /TARGETS/: SPANet assignment targets for reconstruction
    - /source/: PasswdABC format (e, pt, eta, phi per particle)
    - /EventVars/: PasswdABC event variables (normweight)
    - Legacy format for backward compatibility
    """
    import gzip
    import numpy as np
    import h5py
    from tqdm import tqdm

    try:
        import pylhe
    except ImportError:
        print("Error: pylhe not installed")
        sys.exit(1)

    print(f"  Reading LHE file: {lhe_file}")

    # Count events first
    events_list = list(pylhe.read_lhe_with_attributes(str(lhe_file)))
    n_events = len(events_list)
    print(f"  Found {n_events} events")

    # Set up HDF5 output
    max_jets = config.output.max_jets_per_event
    max_particles = config.output.max_particles_per_jet

    # Legacy format features
    PARTICLE_FEATURES = ["pt_rel", "eta_rel", "phi_rel", "energy", "pdg_id"]
    JET_FEATURES = ["pt", "eta", "phi", "mass", "n_constituents", "parent_pdg", "is_signal"]
    EVENT_FEATURES = ["n_jets", "met_x", "met_y", "met_pt", "ht", "n_signal", "weight"]

    # Signal particles for RPV gluino
    signal_pdg_ids = {1000021}  # gluino (abs value)

    # Check if this is an RPV gluino process
    is_rpv_gluino = "1000021" in config.process.process_string or "go" in config.process.process_string.lower()

    with h5py.File(output_path, "w") as f:
        compression = config.output.compression

        # =====================================================================
        # SPANet Format: /INPUTS/Jets/
        # =====================================================================
        inputs_jets = f.create_group("INPUTS/Jets")

        spanet_mask = inputs_jets.create_dataset(
            "MASK", shape=(n_events, max_jets), dtype=bool, compression=compression
        )
        spanet_pt = inputs_jets.create_dataset(
            "pt", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        spanet_eta = inputs_jets.create_dataset(
            "eta", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        spanet_phi = inputs_jets.create_dataset(
            "phi", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        spanet_mass = inputs_jets.create_dataset(
            "mass", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        spanet_btag = inputs_jets.create_dataset(
            "btag", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )

        # =====================================================================
        # SPANet Targets: /TARGETS/ for jet-to-particle assignment
        # For RPV gluino: g1 -> j1 j2 j3, g2 -> j4 j5 j6
        # =====================================================================
        targets = f.create_group("TARGETS")

        if is_rpv_gluino:
            # Gluino 1 decay products (3 jets)
            g1_group = targets.create_group("g1")
            g1_j1 = g1_group.create_dataset(
                "j1", shape=(n_events,), dtype=np.int32, compression=compression
            )
            g1_j2 = g1_group.create_dataset(
                "j2", shape=(n_events,), dtype=np.int32, compression=compression
            )
            g1_j3 = g1_group.create_dataset(
                "j3", shape=(n_events,), dtype=np.int32, compression=compression
            )

            # Gluino 2 decay products (3 jets)
            g2_group = targets.create_group("g2")
            g2_j1 = g2_group.create_dataset(
                "j1", shape=(n_events,), dtype=np.int32, compression=compression
            )
            g2_j2 = g2_group.create_dataset(
                "j2", shape=(n_events,), dtype=np.int32, compression=compression
            )
            g2_j3 = g2_group.create_dataset(
                "j3", shape=(n_events,), dtype=np.int32, compression=compression
            )

        # =====================================================================
        # PasswdABC Format: /source/ and /EventVars/
        # =====================================================================
        source = f.create_group("source")

        passwd_e = source.create_dataset(
            "e", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        passwd_pt = source.create_dataset(
            "pt", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        passwd_eta = source.create_dataset(
            "eta", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        passwd_phi = source.create_dataset(
            "phi", shape=(n_events, max_jets), dtype=np.float32, compression=compression
        )
        passwd_mask = source.create_dataset(
            "mask", shape=(n_events, max_jets), dtype=bool, compression=compression
        )

        event_vars = f.create_group("EventVars")
        normweight = event_vars.create_dataset(
            "normweight", shape=(n_events,), dtype=np.float32, compression=compression
        )

        # =====================================================================
        # Legacy Format (backward compatibility)
        # =====================================================================
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
        for i, event in enumerate(tqdm(events_list, desc="  Converting")):
            # Get final state particles (status == 1)
            final_particles = [p for p in event.particles if p.status == 1]

            jets_data = np.zeros((max_jets, len(JET_FEATURES)), dtype=np.float32)
            particles_data = np.zeros((max_jets, max_particles, len(PARTICLE_FEATURES)), dtype=np.float32)
            jet_mask = np.zeros(max_jets, dtype=bool)
            particle_mask = np.zeros((max_jets, max_particles), dtype=bool)

            # SPANet/PasswdABC arrays
            jet_pt = np.zeros(max_jets, dtype=np.float32)
            jet_eta = np.zeros(max_jets, dtype=np.float32)
            jet_phi = np.zeros(max_jets, dtype=np.float32)
            jet_mass = np.zeros(max_jets, dtype=np.float32)
            jet_energy = np.zeros(max_jets, dtype=np.float32)
            jet_btag = np.zeros(max_jets, dtype=np.float32)

            # Track parent gluino for each jet (for SPANet targets)
            jet_parent_gluino = np.zeros(max_jets, dtype=np.int32)  # 1 or 2 for gluino index

            met_x, met_y = 0.0, 0.0
            ht = 0.0
            n_signal = 0

            # Build particle ancestry map for truth matching
            # Map particle index to parent indices
            particle_parents = {}
            gluino_indices = []

            for idx, p in enumerate(event.particles):
                if abs(p.id) in signal_pdg_ids:
                    n_signal += 1
                    gluino_indices.append(idx)
                # Store mother indices (1-indexed in LHE, convert to 0-indexed)
                if hasattr(p, 'mother1') and p.mother1 > 0:
                    particle_parents[idx] = (p.mother1 - 1, getattr(p, 'mother2', p.mother1) - 1)

            # Process final state particles as "jets" (parton-level)
            jet_idx = 0
            for p_idx, p in enumerate(final_particles):
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

                if p_tot > 0:
                    eta = np.arctanh(np.clip(pz / p_tot, -0.9999, 0.9999))
                else:
                    eta = 0.0

                phi = np.arctan2(p.py, p.px)
                mass = np.sqrt(max(0, p.e**2 - p_tot**2))

                # Apply cuts
                if pt < config.jets.pt_min or abs(eta) > config.jets.eta_max:
                    continue

                # Check if this is a b-quark (for btag)
                is_b = abs(p.id) == 5

                # Legacy format
                jets_data[jet_idx] = [pt, eta, phi, mass, 1, p.id, 0]
                jet_mask[jet_idx] = True
                particles_data[jet_idx, 0] = [1.0, 0.0, 0.0, p.e, p.id]
                particle_mask[jet_idx, 0] = True

                # SPANet/PasswdABC format
                jet_pt[jet_idx] = pt
                jet_eta[jet_idx] = eta
                jet_phi[jet_idx] = phi
                jet_mass[jet_idx] = mass
                jet_energy[jet_idx] = p.e
                jet_btag[jet_idx] = 1.0 if is_b else 0.0

                ht += pt
                jet_idx += 1

            n_jets = jet_idx
            met_pt = np.sqrt(met_x**2 + met_y**2)

            event_data = np.array([n_jets, met_x, met_y, met_pt, ht, n_signal, 1.0], dtype=np.float32)

            # Write Legacy format
            f["jet_features"][i] = jets_data
            f["particle_features"][i] = particles_data
            f["event_features"][i] = event_data
            f["jet_mask"][i] = jet_mask
            f["particle_mask"][i] = particle_mask

            # Write SPANet format
            spanet_mask[i] = jet_mask
            spanet_pt[i] = jet_pt
            spanet_eta[i] = jet_eta
            spanet_phi[i] = jet_phi
            spanet_mass[i] = jet_mass
            spanet_btag[i] = jet_btag

            # Write PasswdABC format
            passwd_e[i] = jet_energy
            passwd_pt[i] = jet_pt
            passwd_eta[i] = jet_eta
            passwd_phi[i] = jet_phi
            passwd_mask[i] = jet_mask
            normweight[i] = 1.0  # Default weight, can be updated with xsec later

            # Write SPANet targets for RPV gluino
            # At parton level, assign first 3 jets to g1, next 3 to g2
            if is_rpv_gluino:
                # Simple assignment: jets 0,1,2 -> g1, jets 3,4,5 -> g2
                # Use -1 for missing jets
                g1_j1[i] = 0 if n_jets > 0 else -1
                g1_j2[i] = 1 if n_jets > 1 else -1
                g1_j3[i] = 2 if n_jets > 2 else -1
                g2_j1[i] = 3 if n_jets > 3 else -1
                g2_j2[i] = 4 if n_jets > 4 else -1
                g2_j3[i] = 5 if n_jets > 5 else -1

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

        # SPANet metadata
        f.attrs["spanet_inputs"] = ["Jets"]
        f.attrs["spanet_jet_features"] = ["pt", "eta", "phi", "mass", "btag"]
        if is_rpv_gluino:
            f.attrs["spanet_targets"] = ["g1", "g2"]
            f.attrs["spanet_target_structure"] = "g1: j1,j2,j3; g2: j1,j2,j3"

        # PasswdABC metadata
        f.attrs["passwdabc_source_features"] = ["e", "pt", "eta", "phi"]
        f.attrs["passwdabc_event_vars"] = ["normweight"]

        print(f"  Output formats: SPANet (/INPUTS/Jets/), PasswdABC (/source/), Legacy")


if __name__ == "__main__":
    main()
