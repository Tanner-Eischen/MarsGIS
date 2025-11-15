"""Command-line interface for MarsHab."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from marshab import __version__
from marshab.core.data_manager import DataManager
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.core.navigation_engine import NavigationEngine
from marshab.types import BoundingBox
from marshab.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="marshab",
    help="Mars Habitat Site Selection and Rover Navigation System",
    add_completion=False
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
    console.print(f"[bold blue]Downloading {dataset} DEM[/bold blue]")
    
    # Parse ROI
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)
    
    # Download
    try:
        dm = DataManager()
        path = dm.download_dem(dataset, bbox, force=force)
        console.print(f"[green]✓ Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
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
    console.print("[bold blue]Starting terrain analysis[/bold blue]")
    
    # Parse ROI
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as e:
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
        console.print("\n[bold green]✓ Analysis complete[/bold green]\n")
        
        table = Table(title="Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Candidate Sites", str(len(results.sites)))
        table.add_row("Top Site Score", f"{results.top_site_score:.3f}")
        table.add_row("Output Directory", str(output))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Analysis failed: {e}[/red]")
        logger.exception("Analysis failed")
        raise typer.Exit(1)


@app.command()
def navigate(
    site_id: int = typer.Argument(..., help="Target site ID"),
    analysis_dir: Path = typer.Option(Path("data/output"), "--analysis", help="Analysis results directory"),
    start_lat: float = typer.Option(..., help="Start latitude"),
    start_lon: float = typer.Option(..., help="Start longitude"),
    output: Path = typer.Option(Path("waypoints.csv"), "--output", "-o", help="Waypoint output file")
):
    """Generate rover navigation waypoints to target site."""
    console.print("[bold blue]Generating navigation waypoints[/bold blue]")
    
    try:
        engine = NavigationEngine()
        waypoints = engine.plan_to_site(
            site_id=site_id,
            analysis_dir=analysis_dir,
            start_lat=start_lat,
            start_lon=start_lon
        )
        
        # Save waypoints
        waypoints.to_csv(output, index=False)
        
        console.print(f"\n[green]✓ Generated {len(waypoints)} waypoints[/green]")
        console.print(f"[green]✓ Saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Navigation failed: {e}[/red]")
        raise typer.Exit(1)


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
        lat_min, lat_max, lon_min, lon_max = map(float, roi.split(','))
        bbox = BoundingBox(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    except Exception as e:
        console.print(f"[red]Invalid ROI format: {e}[/red]")
        raise typer.Exit(1)
    
    try:
        # 1. Download data
        with console.status("[bold green]Downloading DEM..."):
            dm = DataManager()
            dem_path = dm.download_dem(dataset, bbox)
        console.print("[green]✓ DEM downloaded[/green]")
        
        # 2. Run analysis
        with console.status("[bold green]Analyzing terrain..."):
            pipeline_obj = AnalysisPipeline()
            results = pipeline_obj.run(bbox, dataset=dataset)
        console.print("[green]✓ Terrain analyzed[/green]")
        
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
        console.print("[green]✓ Navigation planned[/green]")
        
        # 4. Save all outputs
        output.mkdir(parents=True, exist_ok=True)
        results.save(output)
        waypoints.to_csv(output / "waypoints.csv", index=False)
        
        console.print(f"\n[bold green]✓ Pipeline complete![/bold green]")
        console.print(f"[green]Results saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Pipeline failed: {e}[/red]")
        logger.exception("Pipeline failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

