"""
Truth-level information extractor for HepMC events.

Extracts particle-level information and performs jet clustering
for ML training data preparation.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .config import PipelineConfig
from .utils.pdg import PDGInfo
from .utils.kinematics import delta_r, delta_phi

logger = logging.getLogger(__name__)


@dataclass
class TruthJet:
    """Truth-level jet information."""

    pt: float
    eta: float
    phi: float
    mass: float
    n_constituents: int
    constituents: np.ndarray  # (N, 4) array of constituent 4-momenta [E, px, py, pz]
    constituent_pdg: np.ndarray  # PDG IDs of constituents
    parent_pdg: int = 0  # PDG ID of parent particle (e.g., gluino)
    parent_idx: int = -1  # Index of parent in event
    is_from_signal: bool = False  # True if from signal decay


@dataclass
class TruthEvent:
    """Truth-level event information."""

    event_number: int
    weight: float
    jets: List[TruthJet] = field(default_factory=list)
    met_x: float = 0.0
    met_y: float = 0.0
    n_signal_particles: int = 0
    signal_masses: List[float] = field(default_factory=list)
    ht: float = 0.0

    @property
    def met(self) -> Tuple[float, float]:
        return (self.met_x, self.met_y)

    @property
    def met_pt(self) -> float:
        return np.sqrt(self.met_x**2 + self.met_y**2)


class TruthExtractor:
    """
    Extract truth-level information from HepMC events.

    Performs jet clustering on final-state particles and
    matches jets to signal particle decay products.
    """

    # Map algorithm names to pyjet power parameter
    ALGO_MAP = {
        "antikt": -1,
        "kt": 1,
        "cambridge": 0,
    }

    def __init__(self, config: PipelineConfig):
        """
        Initialize the truth extractor.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.jet_algo = config.jets.algorithm
        self.jet_radius = config.jets.radius
        self.jet_pt_min = config.jets.pt_min
        self.jet_eta_max = config.jets.eta_max

        # Signal particle PDG IDs (for matching)
        self.signal_pdg_ids = self._get_signal_pdg_ids()

        logger.debug(
            f"TruthExtractor initialized: algo={self.jet_algo}, "
            f"R={self.jet_radius}, pt_min={self.jet_pt_min}"
        )

    def _get_signal_pdg_ids(self) -> List[int]:
        """Get PDG IDs of signal particles from config."""
        # Default to gluino for RPV searches
        signal_ids = [1000021]  # gluino

        # Add any particles defined in config
        for particle in self.config.particles:
            if particle.pdg_id not in signal_ids:
                signal_ids.append(particle.pdg_id)

        return signal_ids

    def extract(self, event) -> TruthEvent:
        """
        Extract truth information from a HepMC event.

        Args:
            event: pyhepmc.GenEvent object

        Returns:
            TruthEvent with extracted information
        """
        # Get event weight
        weight = event.weights[0] if event.weights else 1.0

        # Collect final state particles
        final_particles = []
        final_momenta = []
        final_pdg = []

        for particle in event.particles:
            if particle.status == 1:  # Final state
                p = particle.momentum
                final_particles.append(particle)
                final_momenta.append([p.e, p.px, p.py, p.pz])
                final_pdg.append(particle.pid)

        if not final_particles:
            logger.warning(f"Event {event.event_number}: No final state particles found")
            return TruthEvent(event_number=event.event_number, weight=weight)

        final_momenta = np.array(final_momenta, dtype=np.float64)
        final_pdg = np.array(final_pdg, dtype=np.int32)

        # Separate visible and invisible particles
        invisible_mask = np.array([PDGInfo.is_invisible(pdg) for pdg in final_pdg])
        visible_mask = ~invisible_mask

        # Calculate MET from invisible particles
        invisible_momenta = final_momenta[invisible_mask]
        if len(invisible_momenta) > 0:
            met_x = np.sum(invisible_momenta[:, 1])  # px
            met_y = np.sum(invisible_momenta[:, 2])  # py
        else:
            met_x, met_y = 0.0, 0.0

        # Get visible particles for jet clustering
        visible_momenta = final_momenta[visible_mask]
        visible_pdg = final_pdg[visible_mask]
        visible_particles = [p for p, m in zip(final_particles, visible_mask) if m]

        # Cluster jets
        jets = self._cluster_jets(visible_momenta, visible_pdg, visible_particles)

        # Calculate HT
        ht = sum(jet.pt for jet in jets)

        # Get signal particle information
        signal_particles = []
        for particle in event.particles:
            if abs(particle.pid) in self.signal_pdg_ids:
                signal_particles.append(particle)

        signal_masses = [p.momentum.m() for p in signal_particles]

        # Match jets to signal decay products
        self._match_jets_to_signal(jets, event, signal_particles)

        return TruthEvent(
            event_number=event.event_number,
            weight=weight,
            jets=jets,
            met_x=met_x,
            met_y=met_y,
            n_signal_particles=len(signal_particles),
            signal_masses=signal_masses,
            ht=ht,
        )

    def _cluster_jets(
        self,
        momenta: np.ndarray,
        pdg: np.ndarray,
        particles: List,
    ) -> List[TruthJet]:
        """
        Cluster particles into jets using pyjet.

        Args:
            momenta: (N, 4) array of 4-momenta [E, px, py, pz]
            pdg: (N,) array of PDG IDs
            particles: List of HepMC particles

        Returns:
            List of TruthJet objects
        """
        try:
            from pyjet import cluster
        except ImportError:
            logger.error("pyjet not installed. Cannot cluster jets.")
            raise ImportError("pyjet required for jet clustering")

        if len(momenta) == 0:
            return []

        # Convert to pyjet input format: (pT, eta, phi, mass)
        jet_input = self._convert_to_pt_eta_phi_m(momenta)

        # Get algorithm power parameter
        power = self.ALGO_MAP.get(self.jet_algo, -1)

        # Cluster jets
        sequence = cluster(jet_input, R=self.jet_radius, p=power)

        # Extract jets above threshold
        jets = []
        for jet in sequence.inclusive_jets(ptmin=self.jet_pt_min):
            if abs(jet.eta) > self.jet_eta_max:
                continue

            # Get constituent indices
            constituent_indices = []
            for c in jet.constituents_array():
                # pyjet stores user index
                if hasattr(c, "idx"):
                    constituent_indices.append(c.idx)

            # If no indices available, skip
            if not constituent_indices:
                # Fallback: match by position
                constituent_indices = self._match_constituents_by_position(
                    jet, jet_input
                )

            # Extract constituent information
            if constituent_indices:
                constituents = momenta[constituent_indices]
                constituent_pdg_ids = pdg[constituent_indices]
            else:
                constituents = np.array([])
                constituent_pdg_ids = np.array([])

            truth_jet = TruthJet(
                pt=jet.pt,
                eta=jet.eta,
                phi=jet.phi,
                mass=jet.mass,
                n_constituents=len(constituent_indices),
                constituents=constituents,
                constituent_pdg=constituent_pdg_ids,
            )
            jets.append(truth_jet)

        # Sort by pT (descending)
        jets.sort(key=lambda j: j.pt, reverse=True)

        return jets

    def _convert_to_pt_eta_phi_m(self, momenta: np.ndarray) -> np.ndarray:
        """Convert [E, px, py, pz] to [pT, eta, phi, mass] for pyjet."""
        E = momenta[:, 0]
        px = momenta[:, 1]
        py = momenta[:, 2]
        pz = momenta[:, 3]

        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)

        # Protect against division by zero
        p_safe = np.where(p > 0, p, 1e-10)
        eta = np.arctanh(np.clip(pz / p_safe, -0.9999999, 0.9999999))

        phi = np.arctan2(py, px)

        mass_sq = E**2 - p**2
        mass = np.sqrt(np.maximum(mass_sq, 0))

        return np.column_stack([pt, eta, phi, mass]).astype(np.float64)

    def _match_constituents_by_position(
        self, jet, jet_input: np.ndarray, dr_max: float = 0.01
    ) -> List[int]:
        """Match jet constituents by angular position (fallback method)."""
        indices = []
        for const in jet.constituents():
            for i, inp in enumerate(jet_input):
                dr = np.sqrt(
                    (const.eta - inp[1]) ** 2
                    + delta_phi(const.phi, inp[2]) ** 2
                )
                if dr < dr_max and i not in indices:
                    indices.append(i)
                    break
        return indices

    def _match_jets_to_signal(
        self,
        jets: List[TruthJet],
        event,
        signal_particles: List,
    ) -> None:
        """
        Match jets to signal particle decay products.

        Modifies jets in-place to set parent_pdg, parent_idx, and is_from_signal.
        """
        if not signal_particles:
            return

        # For each signal particle, trace decay products
        for sig_idx, sig_particle in enumerate(signal_particles):
            decay_products = self._get_final_decay_products(sig_particle, event)

            # Match jets based on angular distance to decay products
            for jet in jets:
                if jet.is_from_signal:  # Already matched
                    continue

                for decay_p in decay_products:
                    dp = decay_p.momentum
                    dr = delta_r(jet.eta, jet.phi, dp.eta(), dp.phi())

                    if dr < self.jet_radius:
                        jet.parent_pdg = sig_particle.pid
                        jet.parent_idx = sig_idx
                        jet.is_from_signal = True
                        break

    def _get_final_decay_products(self, particle, event) -> List:
        """Recursively get all final state decay products of a particle."""
        products = []

        if particle.end_vertex is None:
            # This is a final state particle
            if particle.status == 1:
                return [particle]
            return []

        for child in particle.end_vertex.particles_out:
            products.extend(self._get_final_decay_products(child, event))

        return products
