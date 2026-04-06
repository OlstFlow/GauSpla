# Third-Party Notices

## Release Scope

This notice file covers the intended **commercial release package** of GauSpla.

At the time of this audit, the release package is expected to contain only:

- `__init__.py`
- `gsp_*.py`
- `README.md`
- `LICENSE`
- files under `docs/` explicitly whitelisted by the release builder

## Bundled Third-Party Source Code

Current audit status: **no third-party source code is intentionally bundled in the release package**.

The release package is intended to exclude:

- `.tmp/`
- `dev/`
- local reference checkouts
- external binaries, wheel files, and `.blend` reference assets

## External References Reviewed During Audit

These materials may exist locally for comparison or research, but are **not part of the commercial release package**:

- `graphdeco-inria/gaussian-splatting`
  - license model: non-commercial / research-only
  - release policy: no code from this project may be shipped in GauSpla commercial releases without separate permission
- `3dgs-render-blender-addon`
  - local reference checkout in `.tmp/`
  - local license file indicates Apache-2.0
  - release policy: excluded from package; no direct source inclusion assumed by current audit
- `LichtFeld-Studio`
  - local reference checkout in `.tmp/`
  - local license file indicates GPLv3
  - release policy: excluded from package; public-facing branding removed from GauSpla UI/docs

## Format and Technique Compatibility

GauSpla intentionally supports common 3DGS `.ply` attribute conventions such as:

- `f_dc_*`
- `f_rest_*`
- `scale_*`
- `rot_*`
- `opacity`

This is treated as format interoperability and common technique support, not as attribution of bundled source code.

## Audit Follow-Up

If future audits identify direct third-party code inclusion, update this file and add the required notices before shipping a public release.
