"""
Event generation pipeline orchestration.

Coordinates MadGraph5, Pythia8, and post-processing stages
to produce HDF5 output files for ML training.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .card_generator import CardGenerator

logger = logging.getLogger(__name__)


class EventGenerator:
    """
    Orchestrates the full event generation pipeline.

    Stages:
    1. Generate MadGraph configuration cards
    2. Run MadGraph to produce LHE events
    3. Run Pythia8 shower with MLM matching
    4. Process events to HDF5 format
    """

    def __init__(
        self,
        config: PipelineConfig,
        docker_mode: bool = True,
        mg5_executable: str = "mg5_aMC",
    ):
        """
        Initialize the event generator.

        Args:
            config: Pipeline configuration
            docker_mode: If True, use Docker containers for generation
            mg5_executable: Path to MadGraph5 executable (for native mode)
        """
        self.config = config
        self.docker_mode = docker_mode
        self.mg5_executable = mg5_executable
        self.card_generator = CardGenerator(config)

        # Set up output directory
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> Path:
        """
        Run the full event generation pipeline.

        Returns:
            Path to the generated HDF5 file
        """
        logger.info(f"Starting event generation for {self.config.name}")
        logger.info(f"Process: {self.config.process.process_string}")
        logger.info(f"Events: {self.config.generation.num_events}")

        # Step 1: Generate MadGraph cards
        logger.info("Step 1/4: Generating MadGraph cards...")
        cards_dir = self.card_generator.generate_all_cards()
        logger.info(f"Cards generated in {cards_dir}")

        # Step 2: Run MadGraph to generate LHE
        logger.info("Step 2/4: Running MadGraph event generation...")
        lhe_file = self._run_madgraph(cards_dir)
        logger.info(f"LHE file generated: {lhe_file}")

        # Step 3: Run Pythia8 shower
        logger.info("Step 3/4: Running Pythia8 parton shower...")
        hepmc_file = self._run_pythia_shower(lhe_file, cards_dir)
        logger.info(f"HepMC file generated: {hepmc_file}")

        # Step 4: Process to HDF5
        logger.info("Step 4/4: Processing events to HDF5...")
        hdf5_file = self._process_to_hdf5(hepmc_file)
        logger.info(f"HDF5 file generated: {hdf5_file}")

        return hdf5_file

    def generate_cards_only(self) -> Path:
        """
        Generate MadGraph cards without running event generation.

        Useful for dry runs and debugging.

        Returns:
            Path to the directory containing generated cards
        """
        logger.info(f"Generating cards only for {self.config.name}")
        cards_dir = self.card_generator.generate_all_cards()
        logger.info(f"Cards generated in {cards_dir}")
        return cards_dir

    def _run_madgraph(self, cards_dir: Path) -> Path:
        """
        Execute MadGraph to generate LHE events.

        Args:
            cards_dir: Directory containing the configuration cards

        Returns:
            Path to the generated LHE file
        """
        if self.docker_mode:
            return self._run_madgraph_docker(cards_dir)
        else:
            return self._run_madgraph_native(cards_dir)

    def _run_madgraph_docker(self, cards_dir: Path) -> Path:
        """Run MadGraph in Docker container."""
        # Build the Docker command
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{cards_dir.absolute()}:/workdir/cards:ro",
            "-v", f"{self.output_dir.absolute()}:/workdir/output",
        ]

        # Add model directory if using custom UFO model
        models_dir = Path("models")
        if models_dir.exists():
            cmd.extend(["-v", f"{models_dir.absolute()}:/workdir/models:ro"])

        cmd.extend([
            "madgraph-ml-madgraph:latest",
            "/workdir/cards/proc_card.dat"
        ])

        logger.debug(f"Running MadGraph Docker command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"MadGraph output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"MadGraph failed: {e.stderr}")
            raise RuntimeError(f"MadGraph generation failed: {e.stderr}")

        # Find the generated LHE file
        return self._find_lhe_file()

    def _run_madgraph_native(self, cards_dir: Path) -> Path:
        """Run MadGraph natively (without Docker)."""
        # Create MG5 script
        script_path = self.card_generator.generate_mg5_script(cards_dir)

        cmd = [self.mg5_executable, str(script_path)]
        logger.debug(f"Running MadGraph command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(self.output_dir),
            )
            logger.debug(f"MadGraph output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"MadGraph failed: {e.stderr}")
            raise RuntimeError(f"MadGraph generation failed: {e.stderr}")

        return self._find_lhe_file()

    def _find_lhe_file(self) -> Path:
        """Find the generated LHE file in the output directory."""
        # Look for LHE files (compressed and uncompressed)
        patterns = ["**/*.lhe.gz", "**/*.lhe", "**/unweighted_events.lhe.gz"]

        for pattern in patterns:
            lhe_files = list(self.output_dir.glob(pattern))
            if lhe_files:
                # Return the most recently modified file
                return max(lhe_files, key=lambda p: p.stat().st_mtime)

        raise FileNotFoundError(f"No LHE file found in {self.output_dir}")

    def _run_pythia_shower(self, lhe_file: Path, cards_dir: Path) -> Path:
        """
        Run Pythia8 parton shower on LHE file.

        Args:
            lhe_file: Path to the input LHE file
            cards_dir: Directory containing the Pythia8 configuration

        Returns:
            Path to the generated HepMC file
        """
        hepmc_file = self.output_dir / f"{self.config.output.filename_prefix}.hepmc"
        pythia_card = cards_dir / "pythia8_card.dat"

        if self.docker_mode:
            return self._run_pythia_docker(lhe_file, hepmc_file, pythia_card)
        else:
            return self._run_pythia_native(lhe_file, hepmc_file, pythia_card)

    def _run_pythia_docker(
        self, lhe_file: Path, hepmc_file: Path, pythia_card: Path
    ) -> Path:
        """Run Pythia8 in Docker container."""
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{lhe_file.parent.absolute()}:/workdir/lhe:ro",
            "-v", f"{hepmc_file.parent.absolute()}:/workdir/output",
            "-v", f"{pythia_card.parent.absolute()}:/workdir/cards:ro",
            "madgraph-ml-pythia:latest",
            f"/workdir/lhe/{lhe_file.name}",
            f"/workdir/output/{hepmc_file.name}",
            "--config", f"/workdir/cards/{pythia_card.name}"
        ]

        logger.debug(f"Running Pythia8 Docker command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"Pythia8 output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Pythia8 failed: {e.stderr}")
            raise RuntimeError(f"Pythia8 shower failed: {e.stderr}")

        return hepmc_file

    def _run_pythia_native(
        self, lhe_file: Path, hepmc_file: Path, pythia_card: Path
    ) -> Path:
        """Run Pythia8 natively using the shower runner script."""
        from .pythia_runner import run_pythia_shower

        run_pythia_shower(
            lhe_input=str(lhe_file),
            hepmc_output=str(hepmc_file),
            config_file=str(pythia_card),
            num_events=self.config.generation.num_events,
        )

        return hepmc_file

    def _process_to_hdf5(self, hepmc_file: Path) -> Path:
        """
        Process HepMC file to HDF5 format.

        Args:
            hepmc_file: Path to the input HepMC file

        Returns:
            Path to the generated HDF5 file
        """
        from .hdf5_writer import HDF5Writer
        from .truth_extractor import TruthExtractor

        output_file = self.output_dir / f"{self.config.output.filename_prefix}.h5"

        writer = HDF5Writer(self.config)
        extractor = TruthExtractor(self.config)

        writer.process_file(hepmc_file, output_file, extractor)

        return output_file


class PipelineRunner:
    """
    Convenience class for running the full pipeline from a configuration file.
    """

    @staticmethod
    def from_yaml(config_path: str, **overrides) -> EventGenerator:
        """
        Create an EventGenerator from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file
            **overrides: Configuration overrides (e.g., num_events=1000)

        Returns:
            Configured EventGenerator instance
        """
        config = PipelineConfig.from_yaml(config_path)

        # Apply overrides
        if "num_events" in overrides:
            config.generation.num_events = overrides["num_events"]
        if "output_dir" in overrides:
            config.output.output_dir = overrides["output_dir"]
        if "random_seed" in overrides:
            config.generation.random_seed = overrides["random_seed"]

        docker_mode = overrides.get("docker_mode", True)

        return EventGenerator(config, docker_mode=docker_mode)

    @staticmethod
    def run(config_path: str, **overrides) -> Path:
        """
        Run the full pipeline from a configuration file.

        Args:
            config_path: Path to the YAML configuration file
            **overrides: Configuration overrides

        Returns:
            Path to the generated HDF5 file
        """
        generator = PipelineRunner.from_yaml(config_path, **overrides)
        return generator.generate()
