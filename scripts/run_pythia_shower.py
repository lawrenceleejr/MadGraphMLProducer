#!/usr/bin/env python3
"""
Pythia8 parton shower runner for MadGraphMLProducer.

Processes LHE files from MadGraph with MLM jet matching
and outputs HepMC format for downstream processing.
"""

import argparse
import gzip
import sys
from pathlib import Path


def run_pythia_shower(
    lhe_input: str,
    hepmc_output: str,
    config_file: str = None,
    num_events: int = -1,
    verbose: bool = False,
):
    """
    Run Pythia8 parton shower on LHE file with MLM matching.

    Args:
        lhe_input: Path to input LHE file (can be .gz compressed)
        hepmc_output: Path for output HepMC file
        config_file: Path to Pythia8 configuration file
        num_events: Maximum events to process (-1 for all)
        verbose: Print progress information
    """
    try:
        import pythia8
    except ImportError:
        print("Error: pythia8 module not found. Please install Pythia8 with Python bindings.")
        sys.exit(1)

    try:
        import pyhepmc
    except ImportError:
        print("Error: pyhepmc module not found. Please install pyhepmc.")
        sys.exit(1)

    # Initialize Pythia
    pythia = pythia8.Pythia()

    # Read configuration file if provided
    if config_file and Path(config_file).exists():
        pythia.readFile(config_file)
        if verbose:
            print(f"Loaded configuration from {config_file}")

    # Handle compressed LHE files
    lhe_path = Path(lhe_input)
    if lhe_path.suffix == ".gz":
        # Decompress to temporary file
        import tempfile

        with gzip.open(lhe_path, "rt") as f_in:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lhe", delete=False
            ) as f_out:
                f_out.write(f_in.read())
                lhe_input = f_out.name
        if verbose:
            print(f"Decompressed LHE file to {lhe_input}")

    # Configure LHE input
    pythia.readString(f"Beams:LHEF = {lhe_input}")
    pythia.readString("Beams:frameType = 4")

    # Initialize Pythia
    if not pythia.init():
        print("Error: Pythia initialization failed")
        sys.exit(1)

    # Set up HepMC output
    output_path = Path(hepmc_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = pyhepmc.io.WriterAscii(str(output_path))

    # Event loop
    n_accepted = 0
    n_tried = 0
    n_failed = 0

    max_events = num_events if num_events > 0 else float("inf")

    if verbose:
        print(f"Starting event loop (max events: {max_events})")

    while n_accepted < max_events:
        # Try to generate next event
        if not pythia.next():
            n_failed += 1
            # Check if we've reached the end of the LHE file
            if pythia.info.atEndOfFile():
                if verbose:
                    print("Reached end of LHE file")
                break
            continue

        n_tried += 1

        # Convert Pythia event to HepMC
        hepmc_event = pythia_to_hepmc(pythia, n_accepted)

        # Write to output
        writer.write(hepmc_event)
        n_accepted += 1

        # Progress reporting
        if verbose and n_accepted % 1000 == 0:
            eff = n_accepted / n_tried * 100 if n_tried > 0 else 0
            print(f"Processed {n_accepted} events (tried {n_tried}, efficiency {eff:.1f}%)")

    writer.close()

    # Print summary
    pythia.stat()

    if verbose:
        print(f"\nSummary:")
        print(f"  Events accepted: {n_accepted}")
        print(f"  Events tried: {n_tried}")
        print(f"  Events failed: {n_failed}")
        if n_tried > 0:
            print(f"  Matching efficiency: {n_accepted / n_tried * 100:.1f}%")
        print(f"  Output file: {hepmc_output}")


def pythia_to_hepmc(pythia, event_number: int):
    """
    Convert Pythia8 event to HepMC3 format.

    Args:
        pythia: Pythia8 instance with current event
        event_number: Event number for output

    Returns:
        pyhepmc.GenEvent
    """
    import pyhepmc
    import numpy as np

    event = pythia.event
    info = pythia.info

    # Create HepMC event
    hepmc_event = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
    hepmc_event.event_number = event_number

    # Add cross-section information
    xs = pyhepmc.GenCrossSection()
    xs.set_cross_section(info.sigmaGen() * 1e9, info.sigmaErr() * 1e9)  # Convert to pb
    hepmc_event.cross_section = xs

    # Add event weights
    hepmc_event.weights = [info.weight()]

    # Create particle map for vertex connections
    particle_map = {}
    vertex_map = {}

    # First pass: create all particles
    for i in range(event.size()):
        p = event[i]
        if p.id() == 0:  # Skip system particle
            continue

        # Create GenParticle
        momentum = pyhepmc.FourVector(p.px(), p.py(), p.pz(), p.e())
        gp = pyhepmc.GenParticle(momentum, p.id(), p.statusHepMC())
        gp.generated_mass = p.m()

        particle_map[i] = gp

    # Second pass: create vertices and connect particles
    for i in range(event.size()):
        p = event[i]
        if p.id() == 0 or i not in particle_map:
            continue

        gp = particle_map[i]

        # Find or create production vertex
        mother1 = p.mother1()
        mother2 = p.mother2()

        if mother1 > 0:
            # Get vertex key based on mothers
            vertex_key = (mother1, mother2 if mother2 > 0 else mother1)

            if vertex_key not in vertex_map:
                # Create new vertex at mother's end position
                mp = event[mother1]
                pos = pyhepmc.FourVector(
                    mp.xProd(), mp.yProd(), mp.zProd(), mp.tProd()
                )
                vertex = pyhepmc.GenVertex(pos)
                hepmc_event.add_vertex(vertex)
                vertex_map[vertex_key] = vertex

                # Add mother particles as incoming
                if mother1 in particle_map:
                    vertex.add_particle_in(particle_map[mother1])
                if mother2 > 0 and mother2 != mother1 and mother2 in particle_map:
                    vertex.add_particle_in(particle_map[mother2])

            # Add this particle as outgoing from the vertex
            vertex_map[vertex_key].add_particle_out(gp)
        else:
            # Beam particle - add directly to event
            hepmc_event.add_particle(gp)

    return hepmc_event


def main():
    parser = argparse.ArgumentParser(
        description="Run Pythia8 parton shower on MadGraph LHE files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "lhe_input",
        help="Input LHE file (can be .gz compressed)",
    )
    parser.add_argument(
        "hepmc_output",
        help="Output HepMC file path",
    )
    parser.add_argument(
        "--config", "-c",
        help="Pythia8 configuration file",
        default=None,
    )
    parser.add_argument(
        "--num-events", "-n",
        type=int,
        default=-1,
        help="Maximum number of events to process (-1 for all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    run_pythia_shower(
        lhe_input=args.lhe_input,
        hepmc_output=args.hepmc_output,
        config_file=args.config,
        num_events=args.num_events,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
