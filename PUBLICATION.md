# Publication layout

The GitHub-ready source and final reports are at this repository root.
This repository is published privately at https://github.com/TheLittleJimmy/reward-engine.

The original Zenodo package is preserved locally in `zenodo/` and excluded from Git.
It contains nine archives, a package manifest, checksums, and extraction instructions.
The archives are available as assets of the private
[evidence release](https://github.com/TheLittleJimmy/reward-engine/releases/tag/evidence-v1.0.0).
They have not been uploaded to Zenodo. Use `python -m release.restore --archives zenodo`
to restore data without overwriting the repaired source with the historical snapshot.

The release metadata and validation reports describe the original publication snapshot.
Author details and the license for original data and reports remain pending; see
`release/PUBLICATION_STATUS.md` and `release/metadata.json`.
