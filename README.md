# MadGraphMLProducer

Generate particle physics events with MadGraph5 and Pythia8 for machine learning training.

## Overview

MadGraphMLProducer is a pipeline for generating simulated particle physics events optimized for ML applications. It:

1. **Generates events** using MadGraph5_aMC@NLO for matrix element calculation
2. **Showers events** using Pythia8 with MLM jet matching
3. **Produces HDF5 output** with truth-level jet information ready for PyTorch

## Features

- YAML-based configuration for easy process specification
- Docker support for reproducible event generation
- MLM jet matching with up to 2 additional partons
- Configurable jet clustering (anti-kt, kt, Cambridge/Aachen)
- HDF5 output optimized for PyTorch DataLoader
- Built-in PyTorch Dataset classes

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lawrenceleejr/MadGraphMLProducer.git
cd MadGraphMLProducer

# Install the package
pip install -e .

# Download UFO models (for SUSY processes)
./scripts/setup_models.sh
```

### Generate Events

```bash
# Generate RPV gluino events (default configuration)
madgraph-ml generate configs/examples/gluino_rpv_1tev.yaml

# Generate with custom settings
madgraph-ml generate configs/examples/gluino_rpv_1tev.yaml \
    --num-events 50000 \
    --output-dir ./my_output

# Dry run (generate cards only)
madgraph-ml generate configs/examples/gluino_rpv_1tev.yaml --dry-run
```

### Use in PyTorch

```python
from torch.utils.data import DataLoader
from madgraph_ml_producer.pytorch_dataset import JetDataset

# Load the generated data
dataset = JetDataset(
    'output/gluino_rpv_1tev/gluino_rpv.h5',
    jet_features=['pt', 'eta', 'phi', 'mass'],
    particle_features=['pt_rel', 'eta_rel', 'phi_rel']
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)

# Training loop
for features, labels in loader:
    jets = features['jets']        # (batch, max_jets, 4)
    particles = features['particles']  # (batch, max_jets, max_particles, 3)
    is_signal = labels             # (batch, max_jets)

    # Your training code here...
```

## Configuration

Create a YAML configuration file to define your physics process:

```yaml
name: "my_process"
description: "My physics process"

collider:
  beam1_pdg: 2212       # proton
  beam2_pdg: 2212
  energy_beam1: 6800.0  # GeV (sqrt(s) = 13.6 TeV)
  energy_beam2: 6800.0

process:
  model: "sm"                    # UFO model
  process_string: "p p > t t~"   # MadGraph process
  decay_chain: null              # Optional decay chain
  extra_partons: 2               # For jet matching (0-2)

particles:
  - pdg_id: 6
    mass: 172.5

matching:
  enabled: true
  scheme: "mlm"
  xqcut: 30.0
  qcut: 40.0

generation:
  num_events: 100000
  random_seed: 12345

jets:
  algorithm: "antikt"
  radius: 0.4
  pt_min: 25.0
  eta_max: 2.5

output:
  output_dir: "./output"
  save_hdf5: true
  compression: "gzip"
```

## RPV Gluino Example

The default example generates RPV SUSY gluino pair production at the LHC:

- **Process**: pp → g̃g̃ with g̃ → jjj (RPV decay)
- **Center of mass energy**: 13.6 TeV
- **Gluino mass**: 1 TeV
- **Jet matching**: MLM with up to 2 additional partons

Each event produces 6 signal jets from gluino decays plus potential ISR/FSR jets.

## HDF5 Output Format

The output HDF5 file contains:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `jet_features` | (N, max_jets, 7) | pt, eta, phi, mass, n_const, parent_pdg, is_signal |
| `particle_features` | (N, max_jets, max_particles, 5) | pt_rel, eta_rel, phi_rel, energy, pdg_id |
| `event_features` | (N, 7) | n_jets, met_x, met_y, met_pt, ht, n_signal, weight |
| `jet_mask` | (N, max_jets) | Valid jet mask |
| `particle_mask` | (N, max_jets, max_particles) | Valid particle mask |

## Docker Usage

Build and run with Docker Compose:

```bash
cd docker

# Build images
docker-compose build

# Run full pipeline
docker-compose run --rm pipeline /workdir/configs/examples/gluino_rpv_1tev.yaml
```

## CLI Commands

```bash
# Generate events
madgraph-ml generate <config.yaml> [options]

# Generate cards only
madgraph-ml make-cards <config.yaml> -o <output_dir>

# Process HepMC to HDF5
madgraph-ml process-hepmc <input.hepmc> <output.h5>

# Inspect HDF5 file
madgraph-ml inspect <file.h5>

# Validate configuration
madgraph-ml validate-config <config.yaml>

# List examples
madgraph-ml list-examples
```

## Requirements

### Python Dependencies
- numpy
- h5py
- pyhepmc
- pylhe
- pyjet
- pydantic
- jinja2
- click
- pyyaml
- tqdm

### Optional
- torch (for PyTorch Dataset classes)
- pythia8 (for native Pythia8 running)

### Docker Images
- `scailfin/madgraph5-amc-nlo:mg5_amc3.5.1`
- `matthewfeickert/pythia-python:pythia8.310`

## Project Structure

```
MadGraphMLProducer/
├── configs/              # Configuration files
│   └── examples/         # Example configurations
├── docker/               # Docker setup
├── cards/
│   └── templates/        # Jinja2 card templates
├── models/               # UFO models (downloaded)
├── src/
│   └── madgraph_ml_producer/
│       ├── config.py           # Configuration system
│       ├── card_generator.py   # Card generation
│       ├── event_generator.py  # Pipeline orchestration
│       ├── truth_extractor.py  # Jet clustering
│       ├── hdf5_writer.py      # HDF5 output
│       ├── pytorch_dataset.py  # PyTorch integration
│       └── cli.py              # Command-line interface
├── scripts/              # Utility scripts
└── output/               # Generated events
```

## License

MIT License
