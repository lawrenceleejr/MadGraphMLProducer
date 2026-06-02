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

## RPV Gluino Samples

Two ready-to-use configurations for R-parity violating gluino pair production:

### Sample 1: Direct RPV Decay (Off-shell squarks)

Gluino decays directly to 3 quarks via RPV coupling: g̃ → uds

```bash
# Generate 10k events (copy-paste this single command)
./run -c configs/examples/gluino_rpv_1tev_uds.yaml -n 10000 --shower off -o gluino_offshell_10k.h5
```

- **Topology**: pp → g̃g̃, g̃ → u d s (6 jets per event)
- **Gluino mass**: 1 TeV
- **√s**: 13.6 TeV (LHC Run 3)

### Sample 2: On-shell Squark Cascade

Gluino decays through an on-shell squark: g̃ → u ũ*, ũ* → d̄ s̄

```bash
# Generate 10k events (copy-paste this single command)
./run -c configs/examples/gluino_rpv_onshell_squark.yaml -n 10000 --shower off -o gluino_onshell_10k.h5
```

- **Topology**: pp → g̃g̃, g̃ → u ũ*, ũ* → d s (6 jets per event, 4 from RPV vertex)
- **Gluino mass**: 1 TeV
- **Squark mass**: 800 GeV
- **√s**: 13.6 TeV (LHC Run 3)

### Quick Comparison

| Sample | Decay | Jets | Intermediate |
|--------|-------|------|--------------|
| Off-shell | g̃ → uds | 6 (3+3) | Virtual squark |
| On-shell | g̃ → u ũ* → u d s | 6 (3+3) | Real 800 GeV squark |

## High-Multiplicity QCD Multijets

Two configurations for generating pure-QCD multijet events with the highest
practical jet multiplicity. The number of hard jets is set in `process_string`
(e.g. `p p > j j j j j j`) — it is **not** limited by `extra_partons` (that
field only adds MLM-matched jets, capped at 2).

### Maximum multiplicity — exclusive six-gluon production

`g g > g g g g g g` — a gluon-gluon initial state producing **exactly six
gluons** at the matrix-element level. Pure glue (gluon initial *and* final
state) makes this the most tractable 2→6 QCD process: one subprocess, no
quark-flavor combinatorics. "Exclusive" = no matching, no extra partons.

```bash
# parton-level, literally exactly six jets per event
./run -c configs/examples/qcd_multijet_max.yaml -n 1000 --shower off -o qcd_6gluon.h5

# or with the Pythia8 shower on top of the six hard gluons
./run -c configs/examples/qcd_multijet_max.yaml -n 1000 -o qcd_6gluon.h5
```

> ⚠️ `gg → 6g` has ~34,300 Feynman diagrams. Diagram generation, phase-space
> integration, and unweighting are all slow and RAM-hungry (expect hours and
> several GB). Start with a small `-n` for a first run, then scale up.

Both a `ptj` cut (soft regulator) **and** a `drjj` cut (collinear regulator)
are required for a finite cross section at fixed order; the config sets both.

### MLM-merged inclusive multijet (recommended for large samples)

Merges the 2-, 3-, and 4-jet matrix elements with MLM matching and lets the
parton shower fill in the rest. Runs at scale (10k+ events) and gives a high,
consistently-described jet-multiplicity tail with no double counting.

```bash
./run -c configs/examples/qcd_multijet_matched.yaml -n 10000 -o qcd_multijet.h5
```

| Config | Approach | ME jets | Cost | Use when |
|--------|----------|---------|------|----------|
| `qcd_multijet_max` | Exclusive `gg → 6g` | exactly 6 | Very high | You need the highest hard-parton multiplicity |
| `qcd_multijet_matched` | MLM merge + shower | 2,3,4 + shower | Moderate | You need a large, realistic inclusive sample |

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
│       ├── gluino_rpv_1tev_uds.yaml      # Direct RPV (off-shell squark)
│       └── gluino_rpv_onshell_squark.yaml # On-shell squark cascade
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
