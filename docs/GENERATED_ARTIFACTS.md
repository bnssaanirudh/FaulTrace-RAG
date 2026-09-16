# Generated-artifact policy

Source control and source releases contain code, configuration, migrations,
tests, and human-authored documentation only.

The following are generated locally and must not be committed or packaged:

- `data/`, including SQLite databases and downloaded benchmark corpora;
- `artifacts/`, `outputs/`, generated figures, tables, and experiment bundles;
- `.next/`, `node_modules/`, `*.tsbuildinfo`, Playwright results, and caches;
- virtual environments, Python caches, coverage files, logs, and local `.env` files.

Every research result intended for citation must instead be exported as a
reproducibility bundle containing the immutable dataset/snapshot identifier,
world record hash, query-spec hash, gold-answer hash, experiment configuration
hash, pipeline/provider/model identifiers, code revision supplied by CI, and
the measured result backing each table or figure.

Use `python scripts/build_release.py` for source releases. The builder uses a
strict source-path allowlist/denylist for tracked and non-ignored untracked
files, then writes a SHA-256 manifest. This prevents newly created source files
from disappearing from a release merely because they have not yet been staged.
