"""
MadGraphMLProducer - Generate particle physics events for ML training

This package provides tools for:
- Configuring MadGraph5 event generation
- Running Pythia8 parton showers with MLM jet matching
- Extracting truth-level jet information
- Writing HDF5 files optimized for PyTorch
"""

__version__ = "0.1.0"

from .config import (
    PipelineConfig,
    ColliderConfig,
    ProcessConfig,
    ParticleConfig,
    MatchingConfig,
    GenerationConfig,
    JetConfig,
    OutputConfig,
)

__all__ = [
    "PipelineConfig",
    "ColliderConfig",
    "ProcessConfig",
    "ParticleConfig",
    "MatchingConfig",
    "GenerationConfig",
    "JetConfig",
    "OutputConfig",
]
