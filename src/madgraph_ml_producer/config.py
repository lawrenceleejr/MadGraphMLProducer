"""
Configuration system for MadGraphMLProducer using Pydantic v2.

Defines all configuration models for the event generation pipeline,
including collider settings, physics processes, jet matching, and output formats.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class ColliderConfig(BaseModel):
    """LHC collider configuration"""

    beam1_pdg: int = Field(default=2212, description="PDG ID for beam 1 (proton=2212)")
    beam2_pdg: int = Field(default=2212, description="PDG ID for beam 2 (proton=2212)")
    energy_beam1: float = Field(default=6800.0, description="Beam 1 energy in GeV")
    energy_beam2: float = Field(default=6800.0, description="Beam 2 energy in GeV")

    @property
    def sqrt_s(self) -> float:
        """Center of mass energy in TeV"""
        return (self.energy_beam1 + self.energy_beam2) / 1000.0

    @property
    def sqrt_s_gev(self) -> float:
        """Center of mass energy in GeV"""
        return self.energy_beam1 + self.energy_beam2


class ProcessConfig(BaseModel):
    """Physics process configuration for MadGraph"""

    model: str = Field(default="RPVMSSM_UFO", description="UFO model name")
    process_string: str = Field(description="MadGraph process string, e.g., 'p p > go go'")
    decay_chain: Optional[str] = Field(
        default=None, description="Decay chain, e.g., '(go > j j j)'"
    )
    extra_partons: int = Field(
        default=2, ge=0, le=2, description="Number of additional partons for jet matching (0-2)"
    )
    excluded_particles: List[str] = Field(
        default_factory=list,
        description="Particles to exclude from diagrams (e.g., squarks)",
    )
    multiparticle_definitions: dict[str, str] = Field(
        default_factory=dict,
        description="Additional multiparticle definitions, e.g., {'sq': 'ul ur dl dr'}",
    )

    @field_validator("extra_partons")
    @classmethod
    def validate_extra_partons(cls, v: int) -> int:
        if not 0 <= v <= 2:
            raise ValueError("extra_partons must be between 0 and 2")
        return v


class ParticleConfig(BaseModel):
    """Particle mass and width configuration"""

    pdg_id: int = Field(description="PDG ID of the particle")
    mass: float = Field(description="Mass in GeV")
    width: Optional[float] = Field(default=None, description="Width in GeV, None=auto-calculate")
    name: Optional[str] = Field(default=None, description="Particle name for reference")

    @field_validator("mass")
    @classmethod
    def validate_mass(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Mass must be non-negative")
        return v


class MatchingConfig(BaseModel):
    """MLM/CKKW-L jet matching configuration"""

    enabled: bool = Field(default=True, description="Enable jet matching")
    scheme: Literal["mlm", "ckkw-l"] = Field(default="mlm", description="Matching scheme")
    xqcut: float = Field(default=30.0, description="kt measure cutoff at ME level in GeV")
    qcut: float = Field(default=40.0, description="Pythia matching scale in GeV")
    max_jet_flavor: int = Field(default=5, description="Max quark flavor considered in jets")
    kt_scheme: int = Field(default=1, description="kt scheme: 1=Durham kt, 0=cone")

    @model_validator(mode="after")
    def validate_matching_scales(self) -> "MatchingConfig":
        if self.enabled and self.qcut < self.xqcut:
            raise ValueError("qcut should typically be >= xqcut for proper matching")
        return self


class GenerationConfig(BaseModel):
    """Event generation settings"""

    num_events: int = Field(default=10000, ge=1, description="Number of events to generate")
    random_seed: int = Field(default=0, description="Random seed (0=automatic)")
    ptj_min: float = Field(default=20.0, ge=0, description="Min jet pT cut in GeV")
    etaj_max: float = Field(default=5.0, ge=0, description="Max jet |eta| cut")
    drjj_min: float = Field(default=0.0, ge=0, description="Min deltaR(j,j) cut")
    ptl_min: float = Field(default=0.0, ge=0, description="Min lepton pT cut in GeV")
    etal_max: float = Field(default=-1.0, description="Max lepton |eta| cut (-1=no cut)")
    use_syst: bool = Field(default=True, description="Enable systematics weights")


class JetConfig(BaseModel):
    """Jet clustering configuration for output processing"""

    algorithm: Literal["antikt", "kt", "cambridge"] = Field(
        default="antikt", description="Jet clustering algorithm"
    )
    radius: float = Field(default=0.4, gt=0, description="Jet radius parameter R")
    pt_min: float = Field(default=25.0, ge=0, description="Min jet pT in GeV for output")
    eta_max: float = Field(default=2.5, ge=0, description="Max jet |eta| for output")
    ghost_area: float = Field(default=0.01, gt=0, description="Ghost area for jet area calculation")


class OutputConfig(BaseModel):
    """Output file configuration"""

    output_dir: str = Field(default="./output", description="Output directory path")
    filename_prefix: str = Field(default="events", description="Prefix for output filenames")
    save_lhe: bool = Field(default=True, description="Save LHE file")
    save_hepmc: bool = Field(default=True, description="Save HepMC file")
    save_hdf5: bool = Field(default=True, description="Save HDF5 file for ML")
    compression: Optional[Literal["gzip", "lzf"]] = Field(
        default="gzip", description="HDF5 compression algorithm"
    )
    max_jets_per_event: int = Field(default=20, ge=1, description="Max jets to store per event")
    max_particles_per_jet: int = Field(
        default=100, ge=1, description="Max constituents per jet"
    )


class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""

    name: str = Field(description="Configuration name/identifier")
    description: Optional[str] = Field(default=None, description="Description of this configuration")

    collider: ColliderConfig = Field(default_factory=ColliderConfig)
    process: ProcessConfig
    particles: List[ParticleConfig] = Field(default_factory=list)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    jets: JetConfig = Field(default_factory=JetConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> "PipelineConfig":
        """Load configuration from a YAML file"""
        import yaml
        from pathlib import Path

        with Path(filepath).open() as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to a YAML file"""
        import yaml
        from pathlib import Path

        with Path(filepath).open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_particle_by_pdg(self, pdg_id: int) -> Optional[ParticleConfig]:
        """Get particle configuration by PDG ID"""
        for p in self.particles:
            if p.pdg_id == pdg_id:
                return p
        return None
