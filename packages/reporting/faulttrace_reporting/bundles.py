"""
Reproducibility Bundle: packages experiment configurations, lock files, metric CSVs, checksum manifests, and integrity verifiers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


class ReproducibilityBundle:
    @staticmethod
    def export_bundle(
        experiment_id: str, spec_dict: dict[str, Any], metrics_df: pd.DataFrame, output_dir: Path
    ) -> Path:
        """Assembles and packages the reproducibility bundle directory with checksum manifests."""
        bundle_dir = output_dir / f"bundle_{experiment_id}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Config JSON
        config_path = bundle_dir / "resolved_config.json"
        config_path.write_text(json.dumps(spec_dict, indent=2))

        # 2. Installed distributions. Reading package module attributes is both
        # incomplete and capable of emitting deprecation warnings.
        lock_path = bundle_dir / "env_packages.lock"
        packages_info = sorted(
            {
                f"{name}=={distribution.version}"
                for distribution in importlib.metadata.distributions()
                if (name := distribution.metadata["Name"])
            }
        )
        lock_path.write_text("\n".join(packages_info) + "\n", encoding="utf-8")

        # 3. Metrics CSV
        metrics_csv = bundle_dir / "metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)

        # 4. Fingerprint actual source and lock state. Never emit fabricated
        # API or prompt hashes in a reproducibility bundle.
        fingerprint_path = bundle_dir / "fingerprints.json"
        repository_root = Path(__file__).resolve().parents[3]
        denied_source_parts = {
            ".next",
            ".next-verify",
            "node_modules",
            "__pycache__",
            "artifacts",
            "outputs",
            "dist",
        }
        source_files = sorted(
            path
            for base in (repository_root / "apps", repository_root / "packages", repository_root / "scripts")
            if base.exists()
            for path in base.rglob("*")
            if path.is_file()
            and not denied_source_parts.intersection(path.relative_to(repository_root).parts)
            and path.suffix in {".py", ".ts", ".tsx", ".js", ".mjs", ".R"}
        )
        source_hasher = hashlib.sha256()
        for source_path in source_files:
            source_hasher.update(source_path.relative_to(repository_root).as_posix().encode("utf-8"))
            source_hasher.update(b"\0")
            source_hasher.update(source_path.read_bytes())

        dependency_lock = repository_root / "requirements.lock.txt"
        fingerprints = {
            "os": sys.platform,
            "platform": platform.platform(),
            "python_version": sys.version,
            "source_tree_sha256": source_hasher.hexdigest(),
            "source_file_count": len(source_files),
            "requirements_lock_sha256": (
                hashlib.sha256(dependency_lock.read_bytes()).hexdigest()
                if dependency_lock.exists()
                else None
            ),
        }
        fingerprint_path.write_text(
            json.dumps(fingerprints, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Capture the exact source bytes used by the run, including uncommitted
        # files. A hash without a recoverable snapshot is insufficient when the
        # working tree is dirty.
        source_snapshot_path = bundle_dir / "source_snapshot.zip"
        with zipfile.ZipFile(source_snapshot_path, "w", zipfile.ZIP_DEFLATED) as snapshot:
            for source_path in source_files:
                snapshot.write(
                    source_path,
                    source_path.relative_to(repository_root).as_posix(),
                )

        # 5. Checksum files
        checksums = {}
        for f_name in [
            "resolved_config.json",
            "env_packages.lock",
            "metrics.csv",
            "fingerprints.json",
            "source_snapshot.zip",
        ]:
            f_path = bundle_dir / f_name
            if f_path.exists():
                sha = hashlib.sha256(f_path.read_bytes()).hexdigest()
                checksums[f_name] = sha

        checksums_path = bundle_dir / "checksums.txt"
        with open(checksums_path, "w") as f:
            for k, v in checksums.items():
                f.write(f"{v}  {k}\n")

        return bundle_dir

    @staticmethod
    def verify_bundle(bundle_dir: Path) -> tuple[bool, list[str]]:
        """Verifies integrity matching hashes registered inside checksums.txt."""
        checksums_path = bundle_dir / "checksums.txt"
        if not checksums_path.exists():
            return False, ["checksums.txt missing"]

        errors = []
        # Parse checksums
        expected_hashes = {}
        with open(checksums_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    expected_hashes[parts[1]] = parts[0]

        for f_name, expected_sha in expected_hashes.items():
            f_path = bundle_dir / f_name
            if not f_path.exists():
                errors.append(f"Missing file: {f_name}")
                continue

            actual_sha = hashlib.sha256(f_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                errors.append(
                    f"Integrity check failed for {f_name}: expected {expected_sha[:12]}, got {actual_sha[:12]}"
                )

        return len(errors) == 0, errors
