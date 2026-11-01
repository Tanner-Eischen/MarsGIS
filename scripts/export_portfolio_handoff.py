"""Export a compact, local-only portfolio handoff without regenerating science."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export(root: Path, output: Path, release_id: str) -> None:
    """Collect allowlisted generated artifacts and distinguish fresh/recorded checks."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", release_id):
        raise ValueError("Invalid handoff release ID")
    if output.exists():
        raise ValueError("Output already exists; retain prior evidence and choose a new directory")
    reference = root / "webui/public/portfolio"
    proof = root / "output/release-2026-10-31-patched/verification.json"
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads(proof.read_text(encoding="utf-8"))
    for name, expected in manifest["outputs_sha256"].items():
        if Path(name).name != name or digest(reference / name) != expected:
            raise ValueError(f"Invalid reference artifact or checksum: {name}")
    checks = []
    logs = {}
    timestamp = datetime.now(UTC).isoformat()
    commands = (
        (
            "independent_static_reference",
            [sys.executable, "-B", "scripts/verify_portfolio_demo.py", str(reference)],
            "python -B scripts/verify_portfolio_demo.py webui/public/portfolio",
        ),
        (
            "export_lint",
            [sys.executable, "-m", "ruff", "check", "scripts/export_portfolio_handoff.py"],
            "python -m ruff check scripts/export_portfolio_handoff.py",
        ),
    )
    for kind, command, display in commands:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"{kind} failed: {result.stdout}\n{result.stderr}")
        logs[f"{kind}.log"] = (result.stdout + result.stderr).replace(str(root), "<repository>")
        checks.append(
            {
                "kind": kind,
                "status": "passed",
                "command": display,
                "environment": f"local {sys.platform}; Python {sys.version.split()[0]}",
                "timestamp_utc": timestamp,
                "evidence_path": f"{kind}.log",
                "freshly_executed": True,
            }
        )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    output.mkdir(parents=True)
    for name in ("report.md", "comparison.png", "routes.geojson"):
        shutil.copyfile(reference / name, output / name)
    for name, content in logs.items():
        (output / name).write_text(content, encoding="utf-8")
    write_json(
        output / "reference-metadata.json",
        {
            "schema_version": 1,
            "reference_manifest_sha256": digest(reference / "manifest.json"),
            "historical_source_revision": manifest["software"]["git_revision"],
            "historical_dirty_worktree": manifest["software"]["dirty"],
            "reference_not_regenerated": True,
            **{key: manifest[key] for key in ("source", "grid", "configuration", "candidates")},
            "reference_output_sha256": manifest["outputs_sha256"],
            "copied_artifacts": ["report.md", "comparison.png", "routes.geojson"],
        },
    )
    write_json(
        output / "validation.json",
        {
            "schema_version": 1,
            "last_verified_utc": timestamp,
            "fresh_checks": checks,
            "reference_output_hashes_checked": len(manifest["outputs_sha256"]),
            "independent_full_graph_shortest_path_check": "passed",
            "recorded_runtime_evidence": {
                "freshly_executed": False,
                "evidence_date": runtime["date"],
                "source_record_sha256": digest(proof),
                "base_revision": runtime["base_git_revision"],
                "release_id": runtime["release_id"],
                "python_tests": runtime["python_tests"],
                "frontend_tests": runtime["frontend_tests"],
                "docker": runtime["docker"],
            },
            "gaps": {
                "fresh_locked_workflow_regeneration": "not_run",
                "browser_interaction": "not_run",
                "patched_live_deployment": "not_run",
                "public_showcase_publication": "pending",
            },
        },
    )
    # Only selected scientific metrics are handed off; scores are not safety probabilities.
    candidate = manifest["candidates"][0]
    handoff = {
        "schema_version": 1,
        "project_id": "marshab",
        "release_id": release_id,
        "source_revision": revision,
        "tested_revision": revision,
        "dirty_worktree": dirty,
        "tested_revision_scope": "Fresh static artifact checker only; dirty source identified by hashes",
        "source_sha256": {
            name: digest(root / name)
            for name in (
                "scripts/verify_portfolio_demo.py",
                "marshab/analysis/verification.py",
                "scripts/export_portfolio_handoff.py",
            )
        },
        "source_access": "private",
        "source_url": None,
        "versioned_evidence_urls": [],
        "public_access_state": "showcase_and_publication_pending",
        "owner_approval_state": "publication_pending",
        "last_verified_utc": timestamp,
        "contribution": "Built observed-DEM terrain processing, relative site scoring, constrained routing, independent export checks and a unified explorer.",
        "checks": {
            "artifact_integrity": "passed",
            "independent_static_reference": "passed",
            "directed_graph_optimality": "passed",
            "runtime": "previously_recorded",
            "browser": "not_run",
            "live_deployment": "not_run",
        },
        "headline_results": [
            {
                "metric": "rank_1_route_length",
                "value": candidate["distance_m"],
                "unit": "m",
                "denominator": "one historical selected-cell route; three candidate routes total",
                "scope": "Jezero CTX reference; 20 m projected analysis grid",
                "selection_protocol": "Highest-scoring eligible cell in largest traversable component; 1.5 km greedy separation",
                "evidence": ["report.md", "routes.geojson", "reference-metadata.json"],
                "limitation": "Planar sampled-grid distance; illustrative constraints, not physical mobility validation",
            }
        ],
        "limitations": [
            "Historical reference retains its original dirty revision; it was not regenerated for this handoff.",
            "The compact handoff omits raster outputs; rerunning the full static check requires the private repository's complete reference.",
            "Fresh checker verifies geometry, masks, scores, accumulated costs and full directed-graph optimality within relative tolerance 1e-6; this is discrete-model validation.",
            "Terrain scores are ROI-relative and do not establish habitat safety, rover energy or travel time.",
            "Global MOLA is coarse and cannot resolve rover-scale hazards.",
            "Recorded local Docker evidence uses 1 GiB and two CPUs, not validated 512 MiB deployment headroom.",
            "Browser/WebGL interactions, public access and patched live deployment remain unverified.",
        ],
        "asset_attributions": [
            {
                "assets": ["comparison.png", "report.md", "routes.geojson"],
                "source": manifest["source"]["citation"],
                "attribution": "Derived from NASA MRO CTX stereo, MOLA-controlled elevation distributed by USGS; generated project visualization and reports",
                "publication_rights_review": "pending; no raw source DEM redistributed",
            }
        ],
        "evidence_references": ["validation.json", "reference-metadata.json", "report.md"],
        "files_sha256": {
            path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write_json(output / "portfolio-evidence.json", handoff)
    for path in output.iterdir():
        if path.suffix in {".json", ".md", ".log"}:
            content = path.read_text(encoding="utf-8")
            if re.search(
                r"(?<![A-Za-z])[A-Za-z]:[\\/]|file://|/Users/|/home/|sk-[A-Za-z0-9]{20}",
                content,
            ):
                raise ValueError(f"Unsafe local-path or credential pattern in {path.name}")
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in sorted(output.iterdir())),
        encoding="utf-8",
    )
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if digest(output / name) != expected:
            raise ValueError(f"Export checksum mismatch: {name}")
    print("Exported compact local handoff; static checker, lint, text scan and hashes passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--release-id", default="marshab-portfolio-evidence-2026-11-01-graph-verified"
    )
    arguments = parser.parse_args()
    export(Path(__file__).resolve().parents[1], arguments.out, arguments.release_id)
