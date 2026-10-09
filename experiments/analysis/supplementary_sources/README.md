# Frozen supplementary analysis sources

These public source snapshots support `python -m experiments.analysis.supplementary`.
Use that wrapper after restoring the Zenodo data. It verifies input hashes, creates an isolated output directory, and adapts only file-identity lookup and absent paper-presentation guards. Statistical definitions and algorithms remain frozen. `prepare_expanded_ranking.py` contains only the pure Jury conversion needed by the completed Jury diagnostic; the stopped study controller remains archived.

The authoritative frozen inputs and dual hashes are in `data/analysis/supplementary/` and `data/provenance/files.jsonl`. These files are not online resume checkpoints.
