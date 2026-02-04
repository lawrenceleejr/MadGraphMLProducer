"""
PyTorch Dataset classes for loading generated HDF5 files.

Provides efficient data loading compatible with PyTorch DataLoader,
including multi-worker support.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union, Callable, Dict, Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# Lazy import torch to allow the module to be imported without torch
def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for JetDataset. "
            "Install with: pip install torch"
        )


class JetDataset:
    """
    PyTorch Dataset for jet physics HDF5 files.

    Handles the multi-worker issue with HDF5 by opening the file
    lazily in __getitem__ rather than in __init__.

    Example usage:
        from torch.utils.data import DataLoader

        dataset = JetDataset(
            'events.h5',
            jet_features=['pt', 'eta', 'phi', 'mass'],
            particle_features=['pt_rel', 'eta_rel', 'phi_rel']
        )
        loader = DataLoader(dataset, batch_size=32, num_workers=4)

        for batch in loader:
            features, labels = batch
            # features['jets'].shape = (32, max_jets, 4)
            # features['particles'].shape = (32, max_jets, max_particles, 3)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        jet_features: Optional[List[str]] = None,
        particle_features: Optional[List[str]] = None,
        event_features: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        signal_only: bool = False,
        return_dict: bool = True,
    ):
        """
        Initialize the JetDataset.

        Args:
            filepath: Path to HDF5 file
            jet_features: List of jet feature names to load (None = all)
            particle_features: List of particle feature names to load (None = all)
            event_features: List of event feature names to load (None = all)
            transform: Optional transform for input features
            target_transform: Optional transform for targets
            signal_only: If True, only include jets from signal particles
            return_dict: If True, return features as dict; else return tuple
        """
        self.torch = _import_torch()

        self.filepath = Path(filepath)
        self.transform = transform
        self.target_transform = target_transform
        self.signal_only = signal_only
        self.return_dict = return_dict

        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.filepath}")

        # Read metadata (safe to do in __init__)
        with h5py.File(self.filepath, "r") as f:
            self.n_events = int(f.attrs["n_events"])
            self.all_jet_features = list(f.attrs["jet_features"])
            self.all_particle_features = list(f.attrs["particle_features"])
            self.all_event_features = list(f.attrs["event_features"])
            self.max_jets = int(f.attrs["max_jets"])
            self.max_particles = int(f.attrs["max_particles"])
            self.config_name = f.attrs.get("config_name", "unknown")

        # Feature selection
        self.jet_features = jet_features or self.all_jet_features
        self.particle_features = particle_features or self.all_particle_features
        self.event_features = event_features or self.all_event_features

        # Compute feature indices for efficient slicing
        self.jet_indices = [
            self.all_jet_features.index(f) for f in self.jet_features
        ]
        self.particle_indices = [
            self.all_particle_features.index(f) for f in self.particle_features
        ]
        self.event_indices = [
            self.all_event_features.index(f) for f in self.event_features
        ]

        # File handle (opened lazily in __getitem__ for multi-worker support)
        self._file = None

        logger.debug(
            f"JetDataset initialized: {self.n_events} events, "
            f"max_jets={self.max_jets}, max_particles={self.max_particles}"
        )

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Any]:
        """
        Get a single event from the dataset.

        Args:
            idx: Event index

        Returns:
            Tuple of (features_dict, labels) if return_dict=True
            Tuple of (jet_features, particle_features, event_features, labels) otherwise
        """
        # Open file if needed (for multi-worker compatibility)
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")

        # Load data with feature selection
        jet_data = self._file["jet_features"][idx][:, self.jet_indices]
        particle_data = self._file["particle_features"][idx][:, :, self.particle_indices]
        event_data = self._file["event_features"][idx][self.event_indices]
        jet_mask = self._file["jet_mask"][idx]
        particle_mask = self._file["particle_mask"][idx]

        # Extract labels (is_signal column from jets)
        is_signal_idx = self.all_jet_features.index("is_signal")
        labels = self._file["jet_features"][idx][:, is_signal_idx]

        # Convert to tensors
        features = {
            "jets": self.torch.from_numpy(jet_data).float(),
            "particles": self.torch.from_numpy(particle_data).float(),
            "event": self.torch.from_numpy(event_data).float(),
            "jet_mask": self.torch.from_numpy(jet_mask),
            "particle_mask": self.torch.from_numpy(particle_mask),
        }

        labels = self.torch.from_numpy(labels).float()

        # Apply transforms
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            labels = self.target_transform(labels)

        if self.return_dict:
            return features, labels
        else:
            return (
                features["jets"],
                features["particles"],
                features["event"],
                features["jet_mask"],
                features["particle_mask"],
                labels,
            )

    def __del__(self):
        """Close file handle on deletion."""
        if self._file is not None:
            self._file.close()

    def get_feature_info(self) -> Dict[str, List[str]]:
        """Get information about available features."""
        return {
            "jet_features": self.jet_features,
            "particle_features": self.particle_features,
            "event_features": self.event_features,
            "all_jet_features": self.all_jet_features,
            "all_particle_features": self.all_particle_features,
            "all_event_features": self.all_event_features,
        }


class FlatJetDataset:
    """
    PyTorch Dataset that flattens jets into individual samples.

    Each sample is a single jet rather than an entire event.
    Useful for jet classification tasks.

    Example usage:
        dataset = FlatJetDataset('events.h5')
        loader = DataLoader(dataset, batch_size=256, num_workers=4)

        for jet_features, particle_features, is_signal in loader:
            # jet_features.shape = (256, n_jet_features)
            # particle_features.shape = (256, max_particles, n_particle_features)
            # is_signal.shape = (256,)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        jet_features: Optional[List[str]] = None,
        particle_features: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        signal_only: bool = False,
        pt_min: float = 0.0,
    ):
        """
        Initialize the FlatJetDataset.

        Args:
            filepath: Path to HDF5 file
            jet_features: List of jet feature names to load
            particle_features: List of particle feature names to load
            transform: Optional transform for input features
            signal_only: If True, only include signal jets
            pt_min: Minimum jet pT threshold
        """
        self.torch = _import_torch()

        self.filepath = Path(filepath)
        self.transform = transform
        self.signal_only = signal_only
        self.pt_min = pt_min

        # Read metadata and build jet index
        with h5py.File(self.filepath, "r") as f:
            self.all_jet_features = list(f.attrs["jet_features"])
            self.all_particle_features = list(f.attrs["particle_features"])
            self.max_particles = int(f.attrs["max_particles"])

            # Build index of (event_idx, jet_idx) pairs
            jet_mask = f["jet_mask"][:]
            jet_features_data = f["jet_features"][:]

            self.jet_index = []
            pt_idx = self.all_jet_features.index("pt")
            is_signal_idx = self.all_jet_features.index("is_signal")

            for evt_idx in range(len(jet_mask)):
                for jet_idx in range(jet_mask.shape[1]):
                    if not jet_mask[evt_idx, jet_idx]:
                        continue

                    pt = jet_features_data[evt_idx, jet_idx, pt_idx]
                    is_signal = jet_features_data[evt_idx, jet_idx, is_signal_idx]

                    # Apply filters
                    if pt < self.pt_min:
                        continue
                    if self.signal_only and not is_signal:
                        continue

                    self.jet_index.append((evt_idx, jet_idx))

        # Feature selection
        self.jet_features = jet_features or self.all_jet_features
        self.particle_features = particle_features or self.all_particle_features

        self.jet_indices = [
            self.all_jet_features.index(f) for f in self.jet_features
        ]
        self.particle_indices = [
            self.all_particle_features.index(f) for f in self.particle_features
        ]

        self._file = None

        logger.debug(
            f"FlatJetDataset initialized: {len(self.jet_index)} jets "
            f"(signal_only={signal_only}, pt_min={pt_min})"
        )

    def __len__(self) -> int:
        return len(self.jet_index)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single jet from the dataset."""
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")

        evt_idx, jet_idx = self.jet_index[idx]

        # Load jet features
        jet_data = self._file["jet_features"][evt_idx, jet_idx, self.jet_indices]
        particle_data = self._file["particle_features"][
            evt_idx, jet_idx, :, self.particle_indices
        ]
        particle_mask = self._file["particle_mask"][evt_idx, jet_idx]

        # Get label
        is_signal_idx = self.all_jet_features.index("is_signal")
        is_signal = self._file["jet_features"][evt_idx, jet_idx, is_signal_idx]

        # Convert to tensors
        jet_tensor = self.torch.from_numpy(jet_data).float()
        particle_tensor = self.torch.from_numpy(particle_data).float()
        mask_tensor = self.torch.from_numpy(particle_mask)
        label = self.torch.tensor(is_signal).float()

        if self.transform:
            jet_tensor, particle_tensor = self.transform(jet_tensor, particle_tensor)

        return jet_tensor, particle_tensor, mask_tensor, label

    def __del__(self):
        if self._file is not None:
            self._file.close()


def collate_jets(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for FlatJetDataset.

    Handles variable-length particle sequences with padding.
    """
    torch = _import_torch()

    jets, particles, masks, labels = zip(*batch)

    return (
        torch.stack(jets),
        torch.stack(particles),
        torch.stack(masks),
        torch.stack(labels),
    )
