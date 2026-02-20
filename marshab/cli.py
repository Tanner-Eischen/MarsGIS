"""Command-line interface for MarsHab."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from marshab import __version__
from marshab.analysis.export import export_suitability_geotiff
from marshab.config import PathfindingStrategy, get_config
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.data_manager import DataManager
from marshab.core.navigation_engine import NavigationEngine
from marshab.mission.scenarios import (
    LandingScenarioParams,
    TraverseScenarioParams,
    run_landing_site_scenario,
    run_rover_traverse_scenario,
)
from marshab.utils.logging import configure_logging, get_logger
from marshab.utils.roi import roi_to_bounding_box

# Disable Rich help to avoid compatibility issues
app = typer.Typer(
    name="marshab",
    help="Mars Habitat Site Selection and Rover Navigation System",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None  # Disable rich formatting to avoid click compatibility issues
)

console = Console()
logger = get_logger(__name__)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        print(f"MarsHab Site Selector v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """MarsHab Site Selector - Mars habitat site selection and rover navigation."""
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, format_type="console")

    # Set config path if provided
    if config_file:
        import os
        os.environ["MARSHAB_CONFIG_PATH"] = str(config_file)


def _download_dem(dataset: str, roi: str, force: bool):
    """Internal function to download DEM data."""
    console.print(f"[bold blue]Downloading {dataset} DEM[/bold blue]")

    # Parse ROI
    try:
        bbox = roi_to_bounding_box(roi)
    except (ValueError, TypeError) as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)

    # Download
    try:
        dm = DataManager()
        path = dm.download_dem(dataset, bbox, force=force)
        # Use ASCII-safe characters for Windows compatibility
        console.print(f"[green]Success! Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error: Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset to download (mola/hirise/ctx)"),
    roi: str = typer.Option(
        ...,
        "--roi",
        help="Region of interest as 'lat_min,lat_max,lon_min,lon_max'"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-download")
):
    """Download Mars DEM data for specified region."""
    _download_dem(dataset, roi, force)


def _analyze_terrain(roi: str, dataset: str, output: Path, threshold: float):
    """Internal function to analyze terrain."""
    console.print("[bold blue]Starting terrain analysis[/bold blue]")

    # Parse ROI
    try:
        bbox = roi_to_bounding_box(roi)
    except (ValueError, TypeError) as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)

    # Run analysis
    try:
        pipeline = AnalysisPipeline()
        results = pipeline.run(bbox, dataset=dataset, threshold=threshold)

        # Save results
        output.mkdir(parents=True, exist_ok=True)
        results.save(output)

        # Display summary
        console.print("\n[bold green]Analysis complete![/bold green]\n")

        table = Table(title="Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Candidate Sites", str(len(results.sites)))
        table.add_row("Top Site Score", f"{results.top_site_score:.3f}")
        table.add_row("Output Directory", str(output))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: Analysis failed: {e}[/red]")
        logger.exception("Analysis failed")
        raise typer.Exit(1)


@app.command()
def analyze(
    roi: str = typer.Option(
        ...,
        "--roi",
        help="Region of interest as 'lat_min,lat_max,lon_min,lon_max'"
    ),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    output: Path = typer.Option(Path("data/output"), "--output", "-o", help="Output directory"),
    threshold: float = typer.Option(0.7, "--threshold", help="Suitability threshold")
):
    """Analyze terrain and identify construction sites."""
    _analyze_terrain(roi, dataset, output, threshold)


def _navigate_to_site(site_id: int, analysis_dir: Path, start_lat: float, start_lon: float, output: Path, strategy: Optional[str] = None):
    """Internal function to generate navigation waypoints."""
    console.print("[bold blue]Generating navigation waypoints[/bold blue]")

    try:
        # Update strategy if provided
        if strategy is not None:
            config = get_config()
            try:
                strategy_enum = PathfindingStrategy(strategy.lower())
                config.navigation.strategy = strategy_enum
                console.print(f"[cyan]Using pathfinding strategy: {strategy}[/cyan]")
            except ValueError:
                console.print(f"[yellow]Warning: Invalid strategy '{strategy}', using default[/yellow]")

        engine = NavigationEngine()
        waypoints = engine.plan_to_site(
            site_id=site_id,
            analysis_dir=analysis_dir,
            start_lat=start_lat,
            start_lon=start_lon
        )

        # Save waypoints
        waypoints.to_csv(output, index=False)

        console.print(f"\n[green]Success! Generated {len(waypoints)} waypoints[/green]")
        console.print(f"[green]Saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: Navigation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def navigate(
    site_id: int = typer.Argument(..., help="Target site ID"),
    analysis_dir: Path = typer.Option(Path("data/output"), "--analysis", help="Analysis results directory"),
    start_lat: float = typer.Option(..., help="Start latitude"),
    start_lon: float = typer.Option(..., help="Start longitude"),
    output: Path = typer.Option(Path("waypoints.csv"), "--output", "-o", help="Waypoint output file"),
    strategy: str = typer.Option("balanced", "--strategy", help="Pathfinding strategy: safest, balanced, or direct")
):
    """Generate rover navigation waypoints to target site."""
    _navigate_to_site(site_id, analysis_dir, start_lat, start_lon, output, strategy)


@app.command()
def pipeline(
    roi: str = typer.Option(..., "--roi", help="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    output: Path = typer.Option(Path("data/output"), "--output", "-o", help="Output directory")
):
    """Run complete analysis and navigation pipeline."""
    console.print("[bold blue]Running full MarsHab pipeline[/bold blue]\n")

    # Parse ROI
    try:
        bbox = roi_to_bounding_box(roi)
    except (ValueError, TypeError) as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)

    try:
        # 1. Download data
        with console.status("[bold green]Downloading DEM..."):
            dm = DataManager()
            dm.download_dem(dataset, bbox)
        console.print("[green]Success! DEM downloaded[/green]")

        # 2. Run analysis
        with console.status("[bold green]Analyzing terrain..."):
            pipeline_obj = AnalysisPipeline()
            results = pipeline_obj.run(bbox, dataset=dataset)
        console.print("[green]Success! Terrain analyzed[/green]")

        # 3. Generate navigation
        with console.status("[bold green]Planning navigation..."):
            engine = NavigationEngine()
            # Navigate to top site
            waypoints = engine.plan_to_site(
                site_id=results.top_site_id,
                analysis_dir=output,
                start_lat=(bbox.lat_min + bbox.lat_max) / 2,
                start_lon=(bbox.lon_min + bbox.lon_max) / 2
            )
        console.print("[green]Success! Navigation planned[/green]")

        # 4. Save all outputs
        output.mkdir(parents=True, exist_ok=True)
        results.save(output)
        waypoints.to_csv(output / "waypoints.csv", index=False)

        console.print("\n[bold green]Pipeline complete![/bold green]")
        console.print(f"[green]Results saved to: {output}[/green]")

    except Exception as e:
        console.print(f"\n[red]Error: Pipeline failed: {e}[/red]")
        logger.exception("Pipeline failed")
        raise typer.Exit(1)


# Create mars command group with shorter aliases
mars_app = typer.Typer(
    name="mars",
    help="Mars Habitat Site Selection - Short command aliases",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None
)


@mars_app.callback()
def mars_main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """Mars - Short aliases for MarsHab commands."""
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, format_type="console")

    # Set config path if provided
    if config_file:
        import os
        os.environ["MARSHAB_CONFIG_PATH"] = str(config_file)


@mars_app.command(name="download")
def mars_download(
    dataset: str = typer.Argument(..., help="Dataset to download (mola/hirise/ctx)"),
    roi: str = typer.Option(
        ...,
        "--roi",
        help="Region of interest as 'lat_min,lat_max,lon_min,lon_max'"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-download")
):
    """Download Mars DEM data for specified region."""
    _download_dem(dataset, roi, force)


@mars_app.command(name="terrain")
def terrain(
    roi: str = typer.Option(
        ...,
        "--roi",
        help="Region of interest as 'lat_min,lat_max,lon_min,lon_max'"
    ),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    output: Path = typer.Option(Path("data/output"), "--output", "-o", help="Output directory"),
    threshold: float = typer.Option(0.7, "--threshold", help="Suitability threshold")
):
    """Analyze terrain and identify construction sites."""
    _analyze_terrain(roi, dataset, output, threshold)


@mars_app.command(name="navigation")
def navigation(
    site_id: int = typer.Argument(..., help="Target site ID"),
    analysis_dir: Path = typer.Option(Path("data/output"), "--analysis", help="Analysis results directory"),
    start_lat: float = typer.Option(..., help="Start latitude"),
    start_lon: float = typer.Option(..., help="Start longitude"),
    output: Path = typer.Option(Path("waypoints.csv"), "--output", "-o", help="Waypoint output file"),
    strategy: str = typer.Option("balanced", "--strategy", help="Pathfinding strategy: safest, balanced, or direct")
):
    """Generate rover navigation waypoints to target site."""
    _navigate_to_site(site_id, analysis_dir, start_lat, start_lon, output, strategy)


@app.command()
def run_landing_scenario(
    roi: str = typer.Option(..., "--roi", help="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    preset: str = typer.Option("balanced", "--preset", help="Preset ID"),
    output: Path = typer.Option(Path("data/output"), "--output", "-o", help="Output directory")
):
    """Run landing site selection scenario."""
    try:
        bbox = roi_to_bounding_box(roi)
    except (ValueError, TypeError) as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)

    params = LandingScenarioParams(
        roi=bbox,
        dataset=dataset.lower(),
        preset_id=preset,
        suitability_threshold=0.7
    )

    try:
        result = run_landing_site_scenario(params)
        console.print("\n[green]Scenario complete![/green]")
        console.print(f"[green]Found {len(result.sites)} sites[/green]")
        if result.top_site:
            console.print(f"[green]Top site: {result.top_site.site_id} (score: {result.top_site.suitability_score:.3f})[/green]")
    except Exception as e:
        console.print(f"[red]Error: Scenario failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run_route_scenario(
    start_site: int = typer.Option(..., "--start-site", help="Start site ID"),
    end_site: int = typer.Option(..., "--end-site", help="End site ID"),
    analysis_dir: Path = typer.Option(Path("data/output"), "--analysis", help="Analysis directory"),
    preset: str = typer.Option("balanced", "--preset", help="Route preset ID"),
    output: Path = typer.Option(Path("route_waypoints.csv"), "--output", "-o", help="Output file")
):
    """Run rover traverse planning scenario."""
    params = TraverseScenarioParams(
        start_site_id=start_site,
        end_site_id=end_site,
        analysis_dir=analysis_dir,
        preset_id=preset
    )

    try:
        result = run_rover_traverse_scenario(params)
        console.print("\n[green]Route planned![/green]")
        console.print(f"[green]Distance: {result.route_metrics.total_distance_m:.2f} m[/green]")
        console.print(f"[green]Waypoints: {result.route_metrics.num_waypoints}[/green]")
    except Exception as e:
        console.print(f"[red]Error: Route planning failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export_suitability(
    roi: str = typer.Option(..., "--roi", help="ROI as 'lat_min,lat_max,lon_min,lon_max'"),
    dataset: str = typer.Option("mola", "--dataset", help="Dataset to use"),
    preset: str = typer.Option("balanced", "--preset", help="Preset ID"),
    output: Path = typer.Option(Path("suitability.tif"), "--output", "-o", help="Output GeoTIFF file")
):
    """Export suitability scores as GeoTIFF."""
    try:
        bbox = roi_to_bounding_box(roi)
    except (ValueError, TypeError) as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)

    # Load preset weights
    from marshab.config.preset_loader import PresetLoader
    loader = PresetLoader()
    preset_obj = loader.get_preset(preset, scope="site")
    weights = preset_obj.get_weights_dict() if preset_obj else {}

    # Run analysis to get suitability
    pipeline = AnalysisPipeline()
    results = pipeline.run(bbox, dataset=dataset.lower(), threshold=0.5)

    if results.suitability is None or results.dem is None:
        console.print("[red]Error: Suitability data not available[/red]")
        raise typer.Exit(1)

    try:
        export_path = export_suitability_geotiff(
            roi=bbox,
            dataset=dataset,
            weights=weights,
            output_path=output,
            dem=results.dem,
            suitability=results.suitability
        )
        console.print(f"[green]Exported suitability GeoTIFF to: {export_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error: Export failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
