# MadGraphMLProducer

Generate particle physics events with MadGraph5 and Pythia8 for machine learning training.

## Quick Start (Docker Only)

**Requirements:** Docker only - no other software needed!

```bash
# Clone the repository
git clone https://github.com/lawrenceleejr/MadGraphMLProducer.git
cd MadGraphMLProducer

# Generate events (builds Docker image on first run)
./run

# Output: events.h5 in current directory
```

That's it! The `./run` command handles everything automatically.

## Usage

```bash
# Default: 10k RPV gluino (1 TeV) events at 13.6 TeV LHC
./run

# Custom number of events
./run -n 50000

# Custom output file
./run -o my_events.h5

# Custom configuration
./run -c /absolute/path/to/config.yaml

# Multiple options
./run -n 100000 -o gluino_100k.h5 -s 42

# Show all options
./run --help

# Rebuild Docker image (if you modify the code)
./run --rebuild
```

### All Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --events` | Number of events | 10000 |
| `-o, --output` | Output HDF5 file | events.h5 |
| `-c, --config` | Config YAML file | RPV gluino 1 TeV |
| `-s, --seed` | Random seed | 12345 |
| `-j, --cores` | CPU cores | 4 |
| `-v, --verbose` | Verbose output | off |
| `--keep-lhe` | Keep LHE file | off |
| `--rebuild` | Force rebuild Docker image | off |

## Default Physics Process

The default configuration generates **RPV SUSY gluino pair production**:

- **Process**: pp → g̃g̃ with g̃ → jjj (R-parity violating decay)
- **√s**: 13.6 TeV (LHC Run 3)
- **Gluino mass**: 1 TeV
- **Jet matching**: MLM with up to 2 additional partons
- **Expected jets**: 6 signal jets per event (from gluino decays) + ISR/FSR

## Output Format

The HDF5 output file is ready for PyTorch:

| Dataset | Shape | Features |
|---------|-------|----------|
| `jet_features` | (N, 20, 7) | pt, eta, phi, mass, n_constituents, parent_pdg, is_signal |
| `particle_features` | (N, 20, 100, 5) | pt_rel, eta_rel, phi_rel, energy, pdg_id |
| `event_features` | (N, 7) | n_jets, met_x, met_y, met_pt, ht, n_signal, weight |
| `jet_mask` | (N, 20) | Valid jet mask |
| `particle_mask` | (N, 20, 100) | Valid particle mask |

### Using in PyTorch

```python
from torch.utils.data import DataLoader
from madgraph_ml_producer.pytorch_dataset import JetDataset

# Load the data
dataset = JetDataset('events.h5')
loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)

# Training loop
for features, labels in loader:
    jets = features['jets']           # (batch, 20, 7)
    particles = features['particles'] # (batch, 20, 100, 5)
    is_signal = labels                # (batch, 20)

    # Your model here...
```

## Custom Configuration

Create a YAML file to define your own physics process:

```yaml
name: "my_process"
description: "My custom physics process"

collider:
  beam1_pdg: 2212       # proton
  beam2_pdg: 2212
  energy_beam1: 6800.0  # GeV → √s = 13.6 TeV
  energy_beam2: 6800.0

process:
  model: "sm"                    # UFO model (sm, MSSM, etc.)
  process_string: "p p > t t~"   # MadGraph syntax
  decay_chain: null              # Optional: "(t > w+ b, w+ > l+ vl)"
  extra_partons: 2               # For MLM matching (0-2)

particles:
  - pdg_id: 6
    mass: 172.5                  # Override particle masses

matching:
  enabled: true
  scheme: "mlm"
  xqcut: 30.0
  qcut: 40.0

generation:
  num_events: 10000
  random_seed: 0                 # 0 = automatic

jets:
  algorithm: "antikt"
  radius: 0.4
  pt_min: 25.0
  eta_max: 2.5

output:
  compression: "gzip"
  max_jets_per_event: 20
  max_particles_per_jet: 100
```

Then run:
```bash
./run -c /path/to/my_config.yaml -n 50000 -o my_events.h5
```

## Project Structure

```
MadGraphMLProducer/
├── run                    # ← Single command entry point
├── Dockerfile             # All-in-one Docker image
├── configs/
│   └── examples/
│       └── gluino_rpv_1tev.yaml
├── cards/templates/       # MadGraph/Pythia8 card templates
└── src/madgraph_ml_producer/
    ├── config.py          # Pydantic configuration
    ├── card_generator.py  # Card generation
    ├── hdf5_writer.py     # HDF5 output
    └── pytorch_dataset.py # PyTorch Dataset classes
```

## Advanced Usage

### Inspect Output File

```bash
# If you have Python dependencies installed
pip install h5py numpy
python -c "
import h5py
with h5py.File('events.h5', 'r') as f:
    print('Events:', f.attrs['n_events'])
    print('Datasets:', list(f.keys()))
    for k, v in f.items():
        print(f'  {k}: {v.shape}')
"
```

### Keep Intermediate Files

```bash
./run --keep-lhe  # Saves events.lhe.gz alongside events.h5
```

## How It Works

1. **Card Generation**: Creates MadGraph5 cards from YAML config
2. **MadGraph5**: Generates hard scattering events (matrix element)
3. **Pythia8**: Adds parton shower, hadronization, MLM matching
4. **Post-processing**: Clusters jets, extracts truth info, writes HDF5

All steps run inside a single Docker container based on the official MadGraph5 image.

## License

MIT License
