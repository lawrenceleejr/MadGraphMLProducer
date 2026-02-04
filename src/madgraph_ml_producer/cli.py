"""
Command-line interface for MadGraphMLProducer.

Provides commands for event generation, card creation, and data processing.
"""

import logging
import sys
from pathlib import Path

import click
import yaml

from .config import PipelineConfig
from .card_generator import CardGenerator
from .event_generator import EventGenerator
from .hdf5_writer import HDF5Inspector


# Configure logging
def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """
    MadGraphMLProducer - Generate particle physics events for ML training.

    This tool orchestrates MadGraph5 and Pythia8 to generate events
    and produces HDF5 output files optimized for PyTorch training.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    help="Override output directory",
)
@click.option(
    "--num-events", "-n",
    type=int,
    help="Override number of events",
)
@click.option(
    "--seed", "-s",
    type=int,
    help="Override random seed",
)
@click.option(
    "--docker/--no-docker",
    default=True,
    help="Use Docker containers (default: True)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate cards only, do not run generation",
)
@click.pass_context
def generate(ctx, config_file, output_dir, num_events, seed, docker, dry_run):
    """
    Generate events from a configuration file.

    CONFIG_FILE should be a YAML file with the pipeline configuration.

    Example:
        madgraph-ml generate configs/examples/gluino_rpv_1tev.yaml
        madgraph-ml generate config.yaml -n 10000 --no-docker
    """
    # Load configuration
    click.echo(f"Loading configuration from {config_file}...")

    try:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
        config = PipelineConfig(**config_dict)
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)

    # Apply overrides
    if output_dir:
        config.output.output_dir = output_dir
    if num_events:
        config.generation.num_events = num_events
    if seed is not None:
        config.generation.random_seed = seed

    # Display configuration
    click.echo(f"\nConfiguration: {config.name}")
    if config.description:
        click.echo(f"Description: {config.description}")
    click.echo(f"Process: {config.process.process_string}")
    if config.process.decay_chain:
        click.echo(f"Decay: {config.process.decay_chain}")
    click.echo(f"Events: {config.generation.num_events}")
    click.echo(f"sqrt(s) = {config.collider.sqrt_s} TeV")
    click.echo(f"Output: {config.output.output_dir}")

    if dry_run:
        click.echo("\n[Dry run] Generating cards only...")
        generator = CardGenerator(config)
        cards_dir = generator.generate_all_cards()
        click.echo(f"\nCards written to: {cards_dir}")
        click.echo("\nGenerated files:")
        for card_file in cards_dir.glob("*.dat"):
            click.echo(f"  - {card_file.name}")
        return

    # Run full pipeline
    click.echo(f"\nStarting event generation (docker={docker})...")

    try:
        generator = EventGenerator(config, docker_mode=docker)
        output_file = generator.generate()
        click.echo(f"\nGeneration complete!")
        click.echo(f"Output file: {output_file}")
    except Exception as e:
        click.echo(f"\nError during generation: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("make-cards")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="./cards/generated",
    help="Output directory for cards",
)
@click.pass_context
def make_cards(ctx, config_file, output_dir):
    """
    Generate MadGraph and Pythia8 cards from configuration.

    This creates the configuration cards without running event generation.

    Example:
        madgraph-ml make-cards config.yaml -o ./my_cards
    """
    # Load configuration
    try:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
        config = PipelineConfig(**config_dict)
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)

    click.echo(f"Generating cards for: {config.name}")

    generator = CardGenerator(config, output_dir=Path(output_dir))

    if not generator.validate_templates():
        click.echo("Warning: Some templates are missing", err=True)

    cards_dir = generator.generate_all_cards()

    click.echo(f"\nCards written to: {cards_dir}")
    click.echo("\nGenerated files:")
    for card_file in sorted(cards_dir.glob("*.dat")):
        click.echo(f"  - {card_file.name}")


@cli.command("process-hepmc")
@click.argument("hepmc_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Configuration file for jet settings",
)
@click.option(
    "--jet-algo",
    type=click.Choice(["antikt", "kt", "cambridge"]),
    default="antikt",
    help="Jet clustering algorithm",
)
@click.option(
    "--jet-radius", "-R",
    type=float,
    default=0.4,
    help="Jet radius parameter",
)
@click.option(
    "--pt-min",
    type=float,
    default=25.0,
    help="Minimum jet pT (GeV)",
)
@click.pass_context
def process_hepmc(ctx, hepmc_file, output_file, config, jet_algo, jet_radius, pt_min):
    """
    Convert HepMC file to HDF5 format.

    Performs jet clustering and extracts truth-level information.

    Example:
        madgraph-ml process-hepmc events.hepmc output.h5
        madgraph-ml process-hepmc events.hepmc output.h5 -c config.yaml
    """
    from .hdf5_writer import HDF5Writer
    from .truth_extractor import TruthExtractor

    # Load or create configuration
    if config:
        with open(config) as f:
            config_dict = yaml.safe_load(f)
        pipeline_config = PipelineConfig(**config_dict)
    else:
        # Create minimal config with provided options
        from .config import ProcessConfig, JetConfig, OutputConfig

        pipeline_config = PipelineConfig(
            name="hepmc_conversion",
            process=ProcessConfig(process_string="unknown"),
            jets=JetConfig(
                algorithm=jet_algo,
                radius=jet_radius,
                pt_min=pt_min,
            ),
            output=OutputConfig(
                output_dir=str(Path(output_file).parent),
                filename_prefix=Path(output_file).stem,
            ),
        )

    click.echo(f"Processing: {hepmc_file}")
    click.echo(f"Jet clustering: {pipeline_config.jets.algorithm} R={pipeline_config.jets.radius}")

    writer = HDF5Writer(pipeline_config)
    extractor = TruthExtractor(pipeline_config)

    try:
        writer.process_file(Path(hepmc_file), Path(output_file), extractor)
        click.echo(f"\nOutput written to: {output_file}")
    except Exception as e:
        click.echo(f"Error processing file: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("hdf5_file", type=click.Path(exists=True))
def inspect(hdf5_file):
    """
    Inspect an HDF5 output file.

    Shows metadata, dataset shapes, and basic statistics.

    Example:
        madgraph-ml inspect output/events.h5
    """
    HDF5Inspector.print_summary(hdf5_file)


@cli.command("validate-config")
@click.argument("config_file", type=click.Path(exists=True))
def validate_config(config_file):
    """
    Validate a configuration file.

    Checks that all required fields are present and have valid values.

    Example:
        madgraph-ml validate-config config.yaml
    """
    click.echo(f"Validating: {config_file}")

    try:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        config = PipelineConfig(**config_dict)

        click.echo(f"\n{click.style('✓', fg='green')} Configuration is valid!")
        click.echo(f"\nConfiguration summary:")
        click.echo(f"  Name: {config.name}")
        click.echo(f"  Process: {config.process.process_string}")
        click.echo(f"  Model: {config.process.model}")
        click.echo(f"  Events: {config.generation.num_events}")
        click.echo(f"  Matching: {config.matching.scheme if config.matching.enabled else 'disabled'}")

        if config.particles:
            click.echo(f"  Particles:")
            for p in config.particles:
                click.echo(f"    - PDG {p.pdg_id}: {p.mass} GeV")

    except Exception as e:
        click.echo(f"\n{click.style('✗', fg='red')} Configuration error: {e}", err=True)
        sys.exit(1)


@cli.command("list-examples")
def list_examples():
    """
    List available example configurations.
    """
    examples_dir = Path(__file__).parent.parent.parent.parent / "configs" / "examples"

    if not examples_dir.exists():
        click.echo("No examples directory found")
        return

    click.echo("Available example configurations:\n")

    for yaml_file in sorted(examples_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                config_dict = yaml.safe_load(f)

            name = config_dict.get("name", yaml_file.stem)
            desc = config_dict.get("description", "No description")

            click.echo(f"  {click.style(name, bold=True)}")
            click.echo(f"    File: {yaml_file.name}")
            click.echo(f"    {desc}\n")

        except Exception:
            click.echo(f"  {yaml_file.name} (error reading)")


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
