"""
Card generator for MadGraph5 and Pythia8.

Uses Jinja2 templates to generate process, run, parameter, and Pythia8 cards
from the pipeline configuration.
"""

import logging
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class CardGenerator:
    """
    Generate MadGraph5 and Pythia8 configuration cards from templates.

    Uses Jinja2 templates stored in cards/templates/ to generate
    the necessary configuration files for event generation.
    """

    # Default template directory (relative to package)
    DEFAULT_TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "cards" / "templates"

    # Card file names
    CARD_FILES = {
        "proc_card": "proc_card.dat",
        "run_card": "run_card.dat",
        "param_card": "param_card.dat",
        "pythia8_card": "pythia8_card.dat",
    }

    def __init__(
        self,
        config: PipelineConfig,
        template_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the card generator.

        Args:
            config: Pipeline configuration
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory to write generated cards
        """
        self.config = config
        self.template_dir = Path(template_dir) if template_dir else self.DEFAULT_TEMPLATE_DIR
        self.output_dir = Path(output_dir) if output_dir else Path("cards/generated")

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

        logger.debug(f"Card generator initialized with template_dir={self.template_dir}")

    def generate_all_cards(self) -> Path:
        """
        Generate all configuration cards.

        Returns:
            Path to the output directory containing generated cards
        """
        # Create output directory
        config_output_dir = self.output_dir / self.config.name
        config_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate each card
        self.generate_proc_card(config_output_dir)
        self.generate_run_card(config_output_dir)
        self.generate_param_card(config_output_dir)
        self.generate_pythia8_card(config_output_dir)

        logger.info(f"Generated all cards in {config_output_dir}")
        return config_output_dir

    def generate_proc_card(self, output_dir: Optional[Path] = None) -> Path:
        """Generate the MadGraph process card."""
        output_dir = output_dir or self.output_dir
        return self._generate_card("proc_card.dat.j2", "proc_card.dat", output_dir)

    def generate_run_card(self, output_dir: Optional[Path] = None) -> Path:
        """Generate the MadGraph run card."""
        output_dir = output_dir or self.output_dir
        return self._generate_card("run_card.dat.j2", "run_card.dat", output_dir)

    def generate_param_card(self, output_dir: Optional[Path] = None) -> Path:
        """Generate the MadGraph parameter card."""
        output_dir = output_dir or self.output_dir
        return self._generate_card("param_card.dat.j2", "param_card.dat", output_dir)

    def generate_pythia8_card(self, output_dir: Optional[Path] = None) -> Path:
        """Generate the Pythia8 configuration card."""
        output_dir = output_dir or self.output_dir
        return self._generate_card("pythia8_card.dat.j2", "pythia8_card.dat", output_dir)

    def _generate_card(
        self, template_name: str, output_name: str, output_dir: Path
    ) -> Path:
        """
        Generate a single card from template.

        Args:
            template_name: Name of the Jinja2 template file
            output_name: Name for the output file
            output_dir: Directory to write the output

        Returns:
            Path to the generated card file
        """
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            raise FileNotFoundError(f"Template not found: {self.template_dir / template_name}")

        # Render template with config
        content = template.render(config=self.config)

        # Write output
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(content)

        logger.debug(f"Generated {output_path}")
        return output_path

    def generate_mg5_script(self, output_dir: Optional[Path] = None) -> Path:
        """
        Generate a MadGraph5 script to run the full generation.

        This script can be passed directly to mg5_aMC.

        Args:
            output_dir: Directory to write the script

        Returns:
            Path to the generated script
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)

        script_content = f"""# MadGraph5 generation script for {self.config.name}
# Generated by MadGraphMLProducer

# Import the process card
import {output_dir / 'proc_card.dat'}

# Launch the generation
launch {self.config.name}
    shower=Pythia8
    done
    {output_dir / 'param_card.dat'}
    {output_dir / 'run_card.dat'}
    {output_dir / 'pythia8_card.dat'}
    done
"""

        script_path = output_dir / "mg5_script.txt"
        with open(script_path, "w") as f:
            f.write(script_content)

        logger.debug(f"Generated MG5 script: {script_path}")
        return script_path

    def validate_templates(self) -> bool:
        """
        Validate that all required templates exist.

        Returns:
            True if all templates are found, False otherwise
        """
        templates = [
            "proc_card.dat.j2",
            "run_card.dat.j2",
            "param_card.dat.j2",
            "pythia8_card.dat.j2",
        ]

        all_found = True
        for template_name in templates:
            template_path = self.template_dir / template_name
            if not template_path.exists():
                logger.error(f"Missing template: {template_path}")
                all_found = False
            else:
                logger.debug(f"Found template: {template_path}")

        return all_found
