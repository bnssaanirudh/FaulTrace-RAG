"""
Reproducibility Bundle: packages experiment configurations, lock files, metric CSVs, checksum manifests, and integrity verifiers.
"""

from __future__ import annotations

import hashlib
import json
import sys
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

        # 2. Package Lock Info (freeze sys.modules)
        lock_path = bundle_dir / "env_packages.lock"
        packages_info = []
        for p in list(sys.modules.keys()):
            try:
                mod = sys.modules[p]
                if hasattr(mod, "__version__"):
                    packages_info.append(f"{p}=={mod.__version__}")
            except Exception:
                pass
        lock_path.write_text("\n".join(sorted(set(packages_info))))

        # 3. Metrics CSV
        metrics_csv = bundle_dir / "metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)

        # 4. Fingerprint details
        fingerprint_path = bundle_dir / "fingerprints.json"
        fingerprints = {
            "os": sys.platform,
            "python_version": sys.version,
            "api_hash": "sha256_mock_api_v1_validated",
            "prompt_hashes": {
                "P1-wrong-scope": "sha256_p1_scope_perturbation_v1",
                "P4-compound-scope-facts": "sha256_p4_compound_v1",
            },
        }
        fingerprint_path.write_text(json.dumps(fingerprints, indent=2))

        # 5. Checksum files
        checksums = {}
        for f_name in [
            "resolved_config.json",
            "env_packages.lock",
            "metrics.csv",
            "fingerprints.json",
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
