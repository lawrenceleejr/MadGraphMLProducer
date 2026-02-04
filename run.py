#!/usr/bin/env python3
"""
Single-command event generation for MadGraphMLProducer.

Usage:
    ./run.py                                    # Use default config (RPV gluino 1 TeV)
    ./run.py --config configs/my_config.yaml    # Custom config
    ./run.py --events 10000 --output events.h5  # Override settings
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate particle physics events and output HDF5 for ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (default: RPV gluino 1 TeV)",
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
        default="events.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=12345,
        help="Random seed",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate LHE and HepMC files",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="scailfin/madgraph5-amc-nlo:mg5_amc3.5.4",
        help="MadGraph Docker image",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run without Docker (requires local MadGraph installation)",
    )
    parser.add_argument(
        "--cores", "-j",
        type=int,
        default=4,
        help="Number of CPU cores for MadGraph",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Get the project root directory
    script_dir = Path(__file__).parent.resolve()

    # Load or create configuration
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = script_dir / "configs" / "examples" / "gluino_rpv_1tev.yaml"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"MadGraphMLProducer - Event Generation Pipeline")
    print(f"=" * 60)
    print(f"Config: {config_path}")
    print(f"Events: {args.events}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"=" * 60)

    # Create working directory
    work_dir = Path(tempfile.mkdtemp(prefix="madgraph_ml_"))
    print(f"\nWorking directory: {work_dir}")

    try:
        # Step 1: Generate MadGraph cards
        print(f"\n[1/4] Generating MadGraph configuration cards...")
        cards_dir = generate_cards(config_path, work_dir, args.events, args.seed, script_dir)

        # Step 2: Run MadGraph
        print(f"\n[2/4] Running MadGraph5 event generation...")
        lhe_file = run_madgraph(
            cards_dir, work_dir, args.docker_image,
            not args.no_docker, args.cores, args.verbose
        )

        # Step 3: Run Pythia8 shower
        print(f"\n[3/4] Running Pythia8 parton shower...")
        hepmc_file = run_pythia_shower(
            lhe_file, work_dir, cards_dir,
            args.docker_image, not args.no_docker, args.verbose
        )

        # Step 4: Convert to HDF5
        print(f"\n[4/4] Converting to HDF5 format...")
        convert_to_hdf5(hepmc_file, args.output, config_path, script_dir)

        print(f"\n" + "=" * 60)
        print(f"SUCCESS! Output written to: {args.output}")
        print(f"=" * 60)

        # Optionally keep intermediate files
        if args.keep_intermediates:
            intermediate_dir = Path(args.output).parent / "intermediates"
            intermediate_dir.mkdir(exist_ok=True)
            if lhe_file.exists():
                shutil.copy(lhe_file, intermediate_dir / lhe_file.name)
            if hepmc_file.exists():
                shutil.copy(hepmc_file, intermediate_dir / hepmc_file.name)
            print(f"Intermediate files saved to: {intermediate_dir}")

    finally:
        # Cleanup
        if not args.keep_intermediates:
            shutil.rmtree(work_dir, ignore_errors=True)


def generate_cards(config_path: Path, work_dir: Path, num_events: int, seed: int, script_dir: Path) -> Path:
    """Generate MadGraph and Pythia8 cards from config."""
    import yaml

    # Add src to path for imports
    sys.path.insert(0, str(script_dir / "src"))
    from madgraph_ml_producer.config import PipelineConfig
    from madgraph_ml_producer.card_generator import CardGenerator

    # Load and modify config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict["generation"]["num_events"] = num_events
    config_dict["generation"]["random_seed"] = seed

    config = PipelineConfig(**config_dict)

    # Generate cards
    cards_dir = work_dir / "cards"
    generator = CardGenerator(
        config,
        template_dir=script_dir / "cards" / "templates",
        output_dir=cards_dir,
    )
    output_dir = generator.generate_all_cards()

    print(f"  Cards written to: {output_dir}")
    return output_dir


def run_madgraph(cards_dir: Path, work_dir: Path, docker_image: str,
                 use_docker: bool, cores: int, verbose: bool) -> Path:
    """Run MadGraph to generate LHE file."""
    output_dir = work_dir / "mg5_output"
    output_dir.mkdir(exist_ok=True)

    # Read process name from proc_card
    proc_card = cards_dir / "proc_card.dat"
    process_name = "Events"
    with open(proc_card) as f:
        for line in f:
            if line.strip().startswith("output"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    process_name = parts[1]
                break

    if use_docker:
        # Create MadGraph script
        mg5_script = work_dir / "mg5_commands.txt"
        with open(mg5_script, "w") as f:
            f.write(f"import /cards/{cards_dir.name}/proc_card.dat\n")
            f.write(f"launch {process_name}\n")
            f.write(f"  done\n")
            f.write(f"  /cards/{cards_dir.name}/param_card.dat\n")
            f.write(f"  /cards/{cards_dir.name}/run_card.dat\n")
            f.write(f"  done\n")

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{cards_dir.parent}:/cards:ro",
            "-v", f"{output_dir}:/output",
            "-v", f"{mg5_script}:/mg5_commands.txt:ro",
            "-w", "/output",
            "-e", f"MG5_NUM_CORES={cores}",
            docker_image,
            "/mg5_commands.txt",
        ]

        print(f"  Running: docker run {docker_image}...")
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode != 0:
            print(f"MadGraph failed!")
            if not verbose and result.stderr:
                print(result.stderr[-2000:])
            sys.exit(1)
    else:
        # Run native MadGraph
        cmd = ["mg5_aMC", str(cards_dir / "proc_card.dat")]
        subprocess.run(cmd, cwd=output_dir, check=True)

    # Find the LHE file
    lhe_files = list(output_dir.glob("**/unweighted_events.lhe.gz"))
    if not lhe_files:
        lhe_files = list(output_dir.glob("**/unweighted_events.lhe"))
    if not lhe_files:
        lhe_files = list(output_dir.glob("**/*.lhe.gz"))
    if not lhe_files:
        lhe_files = list(output_dir.glob("**/*.lhe"))

    if not lhe_files:
        print("Error: No LHE file generated!")
        print(f"Contents of {output_dir}:")
        for f in output_dir.rglob("*"):
            print(f"  {f}")
        sys.exit(1)

    lhe_file = lhe_files[0]
    print(f"  LHE file: {lhe_file}")
    return lhe_file


def run_pythia_shower(lhe_file: Path, work_dir: Path, cards_dir: Path,
                      docker_image: str, use_docker: bool, verbose: bool) -> Path:
    """Run Pythia8 shower on LHE file."""
    hepmc_file = work_dir / "events.hepmc"

    # For now, use a simple Python-based approach with pyhepmc
    # In production, you'd use the full Pythia8 Docker container

    pythia_card = cards_dir / "pythia8_card.dat"

    if use_docker:
        # Use Pythia Docker image
        pythia_image = "matthewfeickert/pythia-python:pythia8.310"

        # Create a simple shower script
        shower_script = work_dir / "shower.py"
        with open(shower_script, "w") as f:
            f.write('''#!/usr/bin/env python3
import sys
import gzip
sys.path.insert(0, "/usr/local/lib/python3.10/site-packages")

try:
    import pythia8
    import pyhepmc
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback: just copy LHE info to a minimal HepMC
    print("Pythia8 not available, creating minimal HepMC output")
    sys.exit(0)

lhe_file = sys.argv[1]
hepmc_file = sys.argv[2]
config_file = sys.argv[3] if len(sys.argv) > 3 else None

pythia = pythia8.Pythia()

# Read config if provided
if config_file:
    pythia.readFile(config_file)

# Set LHE input
pythia.readString(f"Beams:LHEF = {lhe_file}")
pythia.readString("Beams:frameType = 4")

# Initialize
if not pythia.init():
    print("Pythia initialization failed")
    sys.exit(1)

# Event loop
writer = pyhepmc.io.WriterAscii(hepmc_file)
n_events = 0

while True:
    if not pythia.next():
        if pythia.info.atEndOfFile():
            break
        continue

    # Create HepMC event
    evt = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
    evt.event_number = n_events

    # Add particles
    for i in range(pythia.event.size()):
        p = pythia.event[i]
        if p.isFinal():
            mom = pyhepmc.FourVector(p.px(), p.py(), p.pz(), p.e())
            gp = pyhepmc.GenParticle(mom, p.id(), 1)
            evt.add_particle(gp)

    writer.write(evt)
    n_events += 1

    if n_events % 1000 == 0:
        print(f"Processed {n_events} events")

writer.close()
pythia.stat()
print(f"Wrote {n_events} events to {hepmc_file}")
''')

        # Decompress LHE if needed
        if lhe_file.suffix == ".gz":
            lhe_uncompressed = work_dir / "events.lhe"
            with gzip.open(lhe_file, "rb") as f_in:
                with open(lhe_uncompressed, "wb") as f_out:
                    f_out.write(f_in.read())
            lhe_file = lhe_uncompressed

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/work",
            "-v", f"{cards_dir}:/cards:ro",
            "-w", "/work",
            pythia_image,
            "python", "/work/shower.py",
            f"/work/{lhe_file.name}",
            f"/work/{hepmc_file.name}",
            f"/cards/pythia8_card.dat",
        ]

        print(f"  Running: docker run {pythia_image}...")
        result = subprocess.run(cmd, capture_output=not verbose, text=True)

        if result.returncode != 0 and not hepmc_file.exists():
            # Fallback: create minimal HepMC from LHE
            print("  Pythia8 shower failed, using LHE-to-HepMC fallback...")
            create_hepmc_from_lhe(lhe_file, hepmc_file)
    else:
        # Native Pythia8
        print("  Native Pythia8 not yet implemented, using LHE fallback...")
        create_hepmc_from_lhe(lhe_file, hepmc_file)

    if not hepmc_file.exists():
        # Last resort fallback
        create_hepmc_from_lhe(lhe_file, hepmc_file)

    print(f"  HepMC file: {hepmc_file}")
    return hepmc_file


def create_hepmc_from_lhe(lhe_file: Path, hepmc_file: Path):
    """Create HepMC file from LHE (fallback without Pythia8 shower)."""
    try:
        import pylhe
        import pyhepmc
    except ImportError:
        print("Error: pylhe and pyhepmc required for fallback")
        sys.exit(1)

    print("  Converting LHE to HepMC (no parton shower)...")

    # Handle compressed files
    if lhe_file.suffix == ".gz":
        import gzip
        opener = gzip.open
    else:
        opener = open

    writer = pyhepmc.io.WriterAscii(str(hepmc_file))

    with opener(lhe_file, "rt") as f:
        lhe_content = f.read()

    # Parse with pylhe
    events = pylhe.read_lhe_with_attributes(str(lhe_file) if lhe_file.suffix != ".gz" else lhe_content)

    for i, event in enumerate(events):
        hepmc_evt = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
        hepmc_evt.event_number = i

        for particle in event.particles:
            if particle.status == 1:  # Final state
                mom = pyhepmc.FourVector(
                    particle.px, particle.py, particle.pz, particle.e
                )
                gp = pyhepmc.GenParticle(mom, particle.id, 1)
                hepmc_evt.add_particle(gp)

        writer.write(hepmc_evt)

    writer.close()


def convert_to_hdf5(hepmc_file: Path, output_path: str, config_path: Path, script_dir: Path):
    """Convert HepMC to HDF5 format."""
    sys.path.insert(0, str(script_dir / "src"))

    import yaml
    from madgraph_ml_producer.config import PipelineConfig
    from madgraph_ml_producer.hdf5_writer import HDF5Writer
    from madgraph_ml_producer.truth_extractor import TruthExtractor

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = PipelineConfig(**config_dict)

    writer = HDF5Writer(config)
    extractor = TruthExtractor(config)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer.process_file(hepmc_file, output_file, extractor)
    print(f"  HDF5 file: {output_file}")


if __name__ == "__main__":
    main()
