"""
Pythia8 parton shower runner module.

Re-exports the run_pythia_shower function from the scripts directory
for use within the package.
"""

import sys
from pathlib import Path

# Add scripts directory to path for import
_scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

try:
    from run_pythia_shower import run_pythia_shower, pythia_to_hepmc
except ImportError:
    # Fallback if scripts not available - provide stub
    def run_pythia_shower(*args, **kwargs):
        raise ImportError(
            "Pythia8 runner not available. "
            "Please ensure pythia8 and pyhepmc are installed."
        )

    def pythia_to_hepmc(*args, **kwargs):
        raise ImportError("Pythia8 runner not available.")


__all__ = ["run_pythia_shower", "pythia_to_hepmc"]
