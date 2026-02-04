"""
HDF5 writer for truth-level jet information.

Writes events to HDF5 format optimized for PyTorch DataLoader.
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .truth_extractor import TruthExtractor, TruthEvent
from .utils.kinematics import delta_phi

logger = logging.getLogger(__name__)


class HDF5Writer:
    """
    Write truth-level jet information to HDF5 format.

    Output structure (following JetNet conventions):
    - /particle_features: (N_events, N_jets_max, N_particles_max, N_features)
    - /jet_features: (N_events, N_jets_max, N_jet_features)
    - /event_features: (N_events, N_event_features)
    - /jet_mask: (N_events, N_jets_max) - valid jet mask
    - /particle_mask: (N_events, N_jets_max, N_particles_max) - valid particle mask
    """

    # Feature definitions
    PARTICLE_FEATURES = ["pt_rel", "eta_rel", "phi_rel", "energy", "pdg_id"]
    JET_FEATURES = [
        "pt",
        "eta",
        "phi",
        "mass",
        "n_constituents",
        "parent_pdg",
        "is_signal",
    ]
    EVENT_FEATURES = ["n_jets", "met_x", "met_y", "met_pt", "ht", "n_signal", "weight"]

    def __init__(self, config: PipelineConfig):
        """
        Initialize the HDF5 writer.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.max_jets = config.output.max_jets_per_event
        self.max_particles = config.output.max_particles_per_jet
        self.compression = config.output.compression

        logger.debug(
            f"HDF5Writer initialized: max_jets={self.max_jets}, "
            f"max_particles={self.max_particles}, compression={self.compression}"
        )

    def process_file(
        self,
        hepmc_file: Path,
        output_file: Path,
        extractor: TruthExtractor,
    ) -> None:
        """
        Process HepMC file and write to HDF5.

        Args:
            hepmc_file: Path to input HepMC file
            output_file: Path to output HDF5 file
            extractor: TruthExtractor instance for processing events
        """
        try:
            import pyhepmc
        except ImportError:
            raise ImportError("pyhepmc required for reading HepMC files")

        # Count events first
        n_events = self._count_events(hepmc_file)
        logger.info(f"Processing {n_events} events from {hepmc_file}")

        # Create output directory if needed
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize HDF5 file
        with h5py.File(output_file, "w") as f:
            # Create datasets
            self._create_datasets(f, n_events)

            # Process events
            with pyhepmc.open(str(hepmc_file)) as reader:
                for i, event in enumerate(tqdm(reader, total=n_events, desc="Processing")):
                    truth_event = extractor.extract(event)
                    self._write_event(f, i, truth_event)

            # Store metadata
            self._write_metadata(f, n_events)

        logger.info(f"Wrote {n_events} events to {output_file}")

    def _count_events(self, hepmc_file: Path) -> int:
        """Count events in HepMC file."""
        import pyhepmc

        count = 0
        with pyhepmc.open(str(hepmc_file)) as reader:
            for _ in reader:
                count += 1
        return count

    def _create_datasets(self, f: h5py.File, n_events: int) -> None:
        """Create HDF5 datasets with appropriate shapes and compression."""
        compression_opts = {"compression": self.compression} if self.compression else {}

        # Particle features: (events, jets, particles, features)
        particle_shape = (
            n_events,
            self.max_jets,
            self.max_particles,
            len(self.PARTICLE_FEATURES),
        )
        f.create_dataset(
            "particle_features",
            shape=particle_shape,
            dtype=np.float32,
            **compression_opts,
        )

        # Jet features: (events, jets, features)
        jet_shape = (n_events, self.max_jets, len(self.JET_FEATURES))
        f.create_dataset(
            "jet_features",
            shape=jet_shape,
            dtype=np.float32,
            **compression_opts,
        )

        # Event features: (events, features)
        event_shape = (n_events, len(self.EVENT_FEATURES))
        f.create_dataset(
            "event_features",
            shape=event_shape,
            dtype=np.float32,
            **compression_opts,
        )

        # Jet mask: (events, jets)
        f.create_dataset(
            "jet_mask",
            shape=(n_events, self.max_jets),
            dtype=bool,
            **compression_opts,
        )

        # Particle mask: (events, jets, particles)
        f.create_dataset(
            "particle_mask",
            shape=(n_events, self.max_jets, self.max_particles),
            dtype=bool,
            **compression_opts,
        )

    def _write_event(self, f: h5py.File, idx: int, event: TruthEvent) -> None:
        """Write a single event to HDF5 datasets."""
        # Jet features
        n_jets = min(len(event.jets), self.max_jets)
        jet_features = np.zeros((self.max_jets, len(self.JET_FEATURES)), dtype=np.float32)
        jet_mask = np.zeros(self.max_jets, dtype=bool)

        for j, jet in enumerate(event.jets[:n_jets]):
            jet_features[j] = [
                jet.pt,
                jet.eta,
                jet.phi,
                jet.mass,
                jet.n_constituents,
                jet.parent_pdg,
                float(jet.is_from_signal),
            ]
            jet_mask[j] = True

        # Particle features (per jet)
        particle_features = np.zeros(
            (self.max_jets, self.max_particles, len(self.PARTICLE_FEATURES)),
            dtype=np.float32,
        )
        particle_mask = np.zeros((self.max_jets, self.max_particles), dtype=bool)

        for j, jet in enumerate(event.jets[:n_jets]):
            n_const = min(jet.n_constituents, self.max_particles)

            for p in range(n_const):
                if p >= len(jet.constituents):
                    break

                # Extract momentum components [E, px, py, pz]
                E, px, py, pz = jet.constituents[p]

                # Calculate kinematic variables
                pt = np.sqrt(px**2 + py**2)
                p_tot = np.sqrt(px**2 + py**2 + pz**2)
                eta = np.arctanh(np.clip(pz / (p_tot + 1e-10), -0.9999, 0.9999))
                phi = np.arctan2(py, px)

                # Relative coordinates
                pt_rel = pt / jet.pt if jet.pt > 0 else 0
                eta_rel = eta - jet.eta
                phi_rel = delta_phi(phi, jet.phi)

                # PDG ID
                pdg_id = jet.constituent_pdg[p] if p < len(jet.constituent_pdg) else 0

                particle_features[j, p] = [pt_rel, eta_rel, phi_rel, E, pdg_id]
                particle_mask[j, p] = True

        # Event features
        event_features = np.array(
            [
                n_jets,
                event.met_x,
                event.met_y,
                event.met_pt,
                event.ht,
                event.n_signal_particles,
                event.weight,
            ],
            dtype=np.float32,
        )

        # Write to datasets
        f["particle_features"][idx] = particle_features
        f["jet_features"][idx] = jet_features
        f["event_features"][idx] = event_features
        f["jet_mask"][idx] = jet_mask
        f["particle_mask"][idx] = particle_mask

    def _write_metadata(self, f: h5py.File, n_events: int) -> None:
        """Write metadata to HDF5 file."""
        f.attrs["config_name"] = self.config.name
        f.attrs["n_events"] = n_events
        f.attrs["particle_features"] = self.PARTICLE_FEATURES
        f.attrs["jet_features"] = self.JET_FEATURES
        f.attrs["event_features"] = self.EVENT_FEATURES
        f.attrs["max_jets"] = self.max_jets
        f.attrs["max_particles"] = self.max_particles
        f.attrs["jet_algorithm"] = self.config.jets.algorithm
        f.attrs["jet_radius"] = self.config.jets.radius
        f.attrs["jet_pt_min"] = self.config.jets.pt_min
        f.attrs["jet_eta_max"] = self.config.jets.eta_max

        # Store process information
        f.attrs["process_string"] = self.config.process.process_string
        f.attrs["model"] = self.config.process.model
        f.attrs["sqrt_s_tev"] = self.config.collider.sqrt_s

        # Store signal particle masses
        signal_masses = [p.mass for p in self.config.particles]
        f.attrs["signal_masses"] = signal_masses

        logger.debug("Wrote HDF5 metadata")


class HDF5Inspector:
    """Utility class for inspecting HDF5 output files."""

    @staticmethod
    def inspect(filepath: str) -> dict:
        """
        Inspect an HDF5 file and return summary information.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dictionary with file information
        """
        info = {}

        with h5py.File(filepath, "r") as f:
            # Attributes
            info["config_name"] = f.attrs.get("config_name", "unknown")
            info["n_events"] = f.attrs.get("n_events", 0)
            info["jet_algorithm"] = f.attrs.get("jet_algorithm", "unknown")
            info["jet_radius"] = f.attrs.get("jet_radius", 0)
            info["sqrt_s_tev"] = f.attrs.get("sqrt_s_tev", 0)

            # Feature names
            info["particle_features"] = list(f.attrs.get("particle_features", []))
            info["jet_features"] = list(f.attrs.get("jet_features", []))
            info["event_features"] = list(f.attrs.get("event_features", []))

            # Dataset shapes
            info["datasets"] = {}
            for name, ds in f.items():
                info["datasets"][name] = {
                    "shape": ds.shape,
                    "dtype": str(ds.dtype),
                }

            # Basic statistics
            if "jet_mask" in f:
                jet_mask = f["jet_mask"][:]
                info["total_jets"] = int(np.sum(jet_mask))
                info["avg_jets_per_event"] = float(np.mean(np.sum(jet_mask, axis=1)))

            if "event_features" in f:
                event_features = f["event_features"][:]
                # Assuming HT is at index 4
                info["avg_ht"] = float(np.mean(event_features[:, 4]))

        return info

    @staticmethod
    def print_summary(filepath: str) -> None:
        """Print a formatted summary of the HDF5 file."""
        info = HDF5Inspector.inspect(filepath)

        print(f"\n{'=' * 60}")
        print(f"HDF5 File: {filepath}")
        print(f"{'=' * 60}")
        print(f"Configuration: {info['config_name']}")
        print(f"Number of events: {info['n_events']}")
        print(f"sqrt(s) = {info['sqrt_s_tev']} TeV")
        print(f"Jet algorithm: {info['jet_algorithm']} R={info['jet_radius']}")
        print()
        print("Datasets:")
        for name, ds_info in info["datasets"].items():
            print(f"  {name}: {ds_info['shape']} ({ds_info['dtype']})")
        print()
        print("Features:")
        print(f"  Particle: {info['particle_features']}")
        print(f"  Jet: {info['jet_features']}")
        print(f"  Event: {info['event_features']}")
        print()
        if "total_jets" in info:
            print(f"Statistics:")
            print(f"  Total jets: {info['total_jets']}")
            print(f"  Avg jets/event: {info['avg_jets_per_event']:.2f}")
            print(f"  Avg HT: {info['avg_ht']:.1f} GeV")
        print(f"{'=' * 60}\n")
