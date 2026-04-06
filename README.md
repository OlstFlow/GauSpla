# GauSpla

<p align="center">
  <img src="./media/Demo.gif" alt="GauSpla demo" width="960" />
</p>

`GauSpla` is a Blender addon for viewing Gaussian splats from `.ply` files and keeping a Blender object linked to a file on disk for quick reload.

This public bundle contains two parts:

- `Blender_Addon/GauSpla` — the Blender addon
- `LichtFeld_Studio_Plugin/GauSpla_Blender_Sync` — the companion plugin for LichtFeld Studio

## What It Does

### GauSpla (Blender addon)

- imports Gaussian splat `.ply` files into Blender
- keeps the same Blender object linked to the source file
- supports `Auto Sync` so the object reloads when the linked `.ply` changes on disk
- provides `Splats` and `Points` preview modes in the viewport

### GauSpla Blender Sync (LichtFeld Studio plugin)

- adds a one-click sync button in LichtFeld Studio
- exports selected linked splat nodes back into their own `.ply` files
- is intended for the workflow:

`Edit in LichtFeld Studio -> Quick Sync -> Blender reloads the same linked object`

## Installation

### Blender addon

1. Zip the folder `Blender_Addon/GauSpla`
2. In Blender, open:
   `Edit -> Preferences -> Add-ons -> Install...`
3. Select the zip
4. Enable `GauSpla`

You can also install it by dragging the zip into the Blender viewport if your Blender build supports that flow.

### LichtFeld Studio plugin

Copy:

`LichtFeld_Studio_Plugin/GauSpla_Blender_Sync`

to:

`%USERPROFILE%\\.lichtfeld\\plugins\\gauspla_blender_sync`

Then restart LichtFeld Studio.

## Recommended Workflow

1. In Blender, use `Import Linked PLY`
2. Enable `Auto Sync`
3. Open and edit the same `.ply` in LichtFeld Studio
4. Click the sync button in LichtFeld Studio
5. Blender reloads the same linked object

## Warning

⚠️ The LichtFeld Studio sync plugin overwrites the linked `.ply` file in place.

If you do not want to lose the original file, work on a copy or keep a backup.

## Compatibility Note

The companion plugin was built for the LichtFeld Studio plugin API used by `v0.5.x`.

Best-tested path:

- `LichtFeld Studio v0.5.0`

Some stock `v0.5.1` builds may require an upstream toolbar/plugin API patch for exact one-click toolbar behavior.

## License

The Blender addon code in this bundle is released under `GPL-3.0-or-later`.

See:

- `Blender_Addon/GauSpla/LICENSE`
- `Blender_Addon/GauSpla/THIRD_PARTY_NOTICES.md`

## Separate Generic LFS Plugin

If you only need fast `.ply` overwrite inside LichtFeld Studio without Blender, use the separate `PLY Quick Sync` plugin instead of the Blender-oriented companion plugin.
