# XIVUpscaler

Tiered, material-aware upscaling of every vanilla FINAL FANTASY XIV texture — character, world
and UI — into Penumbra mods.

XIVUpscaler is a continuation of [kartoffels123/ffxiv_7_0_toolbox](https://github.com/kartoffels123/ffxiv_7_0_toolbox),
rebuilt around a resumable, manifest-driven pipeline and intended to complement DLSS-NR + SR + MFG
rather than compete with it. The GitHub fork relationship is kept deliberately: see
[Upstream provenance](#upstream-provenance).

**The [wiki](https://github.com/off-cmd/XIVUpscaler/wiki) is the documentation.** This page covers what the
project is, how to install and run it, and the licensing that comes with the models it uses.
Everything else — architecture, the manifest schema, per-role processing, encoding policy,
packaging — lives there, starting at
[Getting Started and Configuration](https://github.com/off-cmd/XIVUpscaler/wiki/Getting-Started-and-Configuration) and
[Pipeline Lifecycle](https://github.com/off-cmd/XIVUpscaler/wiki/Pipeline-Lifecycle).

---

## What it does

The pipeline is a database, not a batch script. Every texture in the game is enumerated once,
classified by family and role, and tracked through its own lifecycle, so a run can be interrupted,
resumed, re-scoped after a classifier change, or re-queued after a game patch without redoing work
that is still valid.

| stage | what happens |
|---|---|
| `plan` | enumerate every vanilla texture, classify it by family and role, write it to `manifest.sqlite` |
| `run` | upscale at the top tier per role on the GPU, resumable, `--budget` in seconds; lower tiers derive from it |
| `pack` | write Penumbra mod folders with a **Tier** option group per family |
| `qa` | contamination, normal-length and colour-drift checks, plus side-by-side comparison sheets |
| `release` | export immutable family/profile releases without re-running inference |

Roles matter because textures are not interchangeable: a normal map must keep unit length, a mask
must not gain colour, UI must stay crisp at native scale. Each role is routed to a different model
and a different post-process — see
[Families, Roles and Classification Policy](https://github.com/off-cmd/XIVUpscaler/wiki/Families-Roles-and-Classification-Policy) and
[Role-Specific Processing Pipelines](https://github.com/off-cmd/XIVUpscaler/wiki/Role-Specific-Processing-Pipelines).

## Requirements

| Requirement | Details |
|---|---|
| Python | **3.11 or 3.12.** `texture2ddecoder` (BC7 decode) publishes no 3.13 wheels |
| GPU | CUDA. `torch` and `torchvision` come from the `cu128` index pinned in `pyproject.toml` |
| texconv | a **release** build from [microsoft/DirectXTex](https://github.com/microsoft/DirectXTex/releases). Run `clarity probe` first — a debug build silently falls back to a CPU BC7 codec ([why this matters](https://github.com/off-cmd/XIVUpscaler/wiki/texconv-Integration-and-Batching)) |
| a game install | read directly. Nothing is downloaded |
| model weights | nine files, obtained separately — see [Upscaling models](#upscaling-models) |
| path list (optional) | a ResLogger `CurrentPathList-<date>.gz` from [rl2.perchbird.dev](https://rl2.perchbird.dev), for paths the index alone does not name |

## Install

```
uv sync
uv run clarity where
```

Or, as a package: `uv pip install xivupscaler` (or `pip install xivupscaler`) gives you a
`clarity` command. An installed copy resolves its project directory from `CLARITY_PROJECT`,
else the directory you run it from.

`clarity where` prints the resolved layout — database, models, texconv, shared tools, path list —
and marks anything missing. Run it before anything else; it answers most setup questions on its own.

Every location is a default derived from the package's position on disk, and every one is
overridden by a `CLARITY_*` environment variable. The full table, and why `CLARITY_SCRATCH` is
worth setting deliberately on a machine where scratch writes matter, are in
[Getting Started and Configuration](https://github.com/off-cmd/XIVUpscaler/wiki/Getting-Started-and-Configuration).

## Runbook

```
uv run clarity plan --chara --icons
uv run clarity estimate
uv run clarity probe
uv run clarity run --budget 3600
uv run clarity pack
uv run clarity qa
```

`requeue`, `reclassify`, `audit`, `fingerprint` and `modup` handle everything after the first pass —
re-queueing failed or stale rows, reclassifying after a policy change, and turning a game patch into
a computed re-upscale set rather than a guess. See
[Maintenance Commands](https://github.com/off-cmd/XIVUpscaler/wiki/Maintenance-Commands), or
[Command-Line Interface](https://github.com/off-cmd/XIVUpscaler/wiki/Command-Line-Interface) for the whole surface.

## Development

```
uv sync                        # runtime + dev tools
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run ty check                # types
uv run pytest --cov            # tests; no game, GPU or weights needed
uv build && uvx twine check dist/*
```

`CONTRIBUTING.md` has the layout, the conventions and the list of things that bite;
[Development, Testing and Tooling](https://github.com/off-cmd/XIVUpscaler/wiki/Development-Testing-and-Tooling) covers the test suite and CI.

---

## Upscaling models

**XIVUpscaler does not redistribute neural-network model weights.** The nine files below are
external dependencies that you obtain separately and place in `analysis-specimens/models/` (or
wherever `CLARITY_MODELS` points). `clarity/models/registry.json` maps each pipeline slot to a
filename; a `registry.json` beside the weights overrides individual slots.

Licenses below were read from each model's own listing in September 2026, not inferred from the
file names.

| slot | file | arch | author | license | source |
|---|---|---|---|---|---|
| `bc1clean` | `1x_BC1-smooth2.pth` | ESRGAN | BlueAmulet | **CC-BY-NC-4.0** | [OpenModelDB](https://openmodeldb.info/models/1x-BC1-smooth2) |
| `normal` | `4x-Normal-RG0-BC7.pth` | ESRGAN | RunDevelopment | CC0-1.0 | [OpenModelDB](https://openmodeldb.info/models/4x-Normal-RG0-BC7) · [rundev-models](https://github.com/RunDevelopment/rundev-models/blob/main/normals/README.md) |
| `normal_bc1` | `4x-Normal-RG0-BC1.pth` | ESRGAN | RunDevelopment | CC0-1.0 | [OpenModelDB](https://openmodeldb.info/models/4x-Normal-RG0-BC1) · [rundev-models](https://github.com/RunDevelopment/rundev-models/blob/main/normals/README.md) |
| `color` | `4x-PBRify_UpscalerV4.pth` | DAT2 | Kim2091 | CC0-1.0 | [OpenModelDB](https://openmodeldb.info/models/4x-PBRify-UpscalerV4) · [release](https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_UpscalerV4) |
| `color_v3` | `4x-PBRify_RPLKSRd_V3.pth` | RealPLKSR_dysample | Kim2091 | CC0-1.0 | [OpenModelDB](https://openmodeldb.info/models/4x-PBRify-RPLKSRd-V3) · [release](https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_RPLKSRd_V3) |
| `mask` | `4x-PBRify_UpscalerSPANV4.pth` | SPAN | Kim2091 | CC0-1.0 | [OpenModelDB](https://openmodeldb.info/models/4x-PBRify-UpscalerSPANV4) · [PBRify_Remix](https://github.com/Kim2091/PBRify_Remix) |
| `face` | `4xFaceUpDAT.pth` | DAT | Helaman (Philip Hofmann) | CC-BY-4.0 | [OpenModelDB](https://openmodeldb.info/models/4x-FaceUpDAT) · [Hugging Face](https://huggingface.co/Phips/4xFaceUpDAT) |
| `skin` | `x1_ITF_SkinDiffDDS_v1.pth` | ESRGAN | intheflesh | **CC-BY-NC-4.0** | [OpenModelDB](https://openmodeldb.info/models/1x-ITF-SkinDiffDDS-v1) |
| `ui` | `4x-UltraSharpV2.safetensors` | DAT2 | Kim2091 | **CC-BY-NC-SA-4.0** | [OpenModelDB](https://openmodeldb.info/models/4x-UltraSharpV2) · [Hugging Face](https://huggingface.co/Kim2091/UltraSharpV2) · [release](https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-UltraSharpV2) |

Every model here was sourced from [OpenModelDB](https://openmodeldb.info), so its listing is given
first on every row and is the licence of record. Mirrors and the authors' own release pages follow
it, for when a download link rots or you want the training notes.

### Three of these are NonCommercial, and it reaches the output

This is the part most easily got wrong, so it is stated plainly rather than left to the table:

* **`1x_BC1-smooth2` (bc1clean)** — CC-BY-NC-4.0
* **`x1_ITF_SkinDiffDDS_v1` (skin)** — CC-BY-NC-4.0
* **`4x-UltraSharpV2` (ui)** — CC-BY-NC-**SA**-4.0

A NonCommercial term is generally understood to restrict what you may do with what the model
produces, not merely with the weights file. Textures generated through those three slots should
therefore be treated as carrying NonCommercial terms, and the UI slot additionally carries
ShareAlike. The remaining six are CC0 or attribution-only.

**Five of the nine are CC0-1.0** and impose no conditions at all — the three Kim2091 PBRify models
(`color`, `color_v3`, `mask`) and both RunDevelopment normals models (`normal`, `normal_bc1`). A
pipeline restricted to those five slots inherits no obligations whatsoever. `face` is
attribution-only.

None of this is legal advice, and XIVUpscaler's own licence does not and cannot override any of it.
Read each model's terms at the links above before distributing anything you generate.

### Third-party content

XIVUpscaler does not distribute FINAL FANTASY XIV game assets or the third-party neural-network
model weights used by the upscaling pipeline. It reads textures from a game installation that you
already own, and it loads model weights that you obtain separately.

Users provide these dependencies themselves. Third-party content remains subject to the copyright,
licence and usage terms of its respective owners. FINAL FANTASY XIV is © SQUARE ENIX CO., LTD.
This project is unaffiliated with and unendorsed by Square Enix.

---

## Upstream provenance

XIVUpscaler is a fork and continuation of
[kartoffels123/ffxiv_7_0_toolbox](https://github.com/kartoffels123/ffxiv_7_0_toolbox). Original
upstream material remains the work of Kartoffels and attributable to them.

**That repository carries no `LICENSE` file.** Its author has, however, stated their permissions
publicly and unambiguously elsewhere:

> "Everything I produce is open source and free use. If you appreciate my work you can donate to my
> kofi, but are in no way obliged to do so."
>
> — [kartoffels' Heliosphere profile](https://heliosphere.app/user/46fzf43g8s34q1x9z9xd8q7xkr)

and, in the **Permissions** field of their mod releases:

> "I don't care"
>
> — e.g. [Kartoffels Upscaled Human Textures](https://heliosphere.app/mod/g1vk6fqpa10813vhjjfc211wzr)

Those statements are strong evidence of intent to permit reuse, including this continuation. They
are **not** a standard software licence: they do not spell out rights to modify, redistribute,
sublicense or commercialise the way MIT, 0BSD or CC0 do, and GitHub cannot detect them.

XIVUpscaler therefore does **not** relicense the inherited material. The `LICENSE` file in this
repository covers the code written for XIVUpscaler and nothing else. If Kartoffels ever adds an
explicit licence to `ffxiv_7_0_toolbox`, that resolves the ambiguity at its source and this section
should be updated to point at it.

## Licensing, in three separate layers

Keeping these apart is the whole point; collapsing them into one claim would be wrong in both
directions.

1. **XIVUpscaler's own code** — [0BSD](LICENSE). Public-domain-equivalent: use, copy, modify and
   distribute for any purpose, with or without fee, no attribution required.
2. **Inherited `ffxiv_7_0_toolbox` material** — Kartoffels' original work, under the stated
   permissions quoted above. Not relicensed here.
3. **Model weights and game assets** — third-party, not distributed by this project, each governed
   by its own terms. See the table above.

## Credits

XIVUpscaler is based on and continues the work of
[kartoffels123/ffxiv_7_0_toolbox](https://github.com/kartoffels123/ffxiv_7_0_toolbox).

If you find this project useful, please consider supporting Kartoffels, whose work made it possible:

☕ **[Buy Kartoffels a coffee](https://ko-fi.com/kartoffels)**

That link supports the upstream author, not this fork's maintainer.

Thanks also to the model authors whose weights this pipeline depends on — **Kim2091**,
**RunDevelopment**, **Helaman** (Philip Hofmann), **BlueAmulet** and **intheflesh** — and to
[OpenModelDB](https://openmodeldb.info) for making their terms findable in the first place.

Supporting tools this project relies on: [DirectXTex](https://github.com/microsoft/DirectXTex)
(texconv), [spandrel](https://github.com/chaiNNer-org/spandrel) for model loading,
[Penumbra](https://github.com/xivdev/Penumbra) as the mod runtime, and
[ResLogger](https://rl2.perchbird.dev) for path lists.
