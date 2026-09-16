# clarity.ffxiv — the FINAL FANTASY XIV file-access layer

The readers and writers for the game's binary formats. Imported as `from clarity import
ffxiv as kb`; the module names are canonical, and `kb.exdfile`, `kb.mtrlfile`, `kb.texfile`
remain as aliases for code written against the earlier `ffxiv-kbtools` names.

| module | what |
|---|---|
| `sqpack` | index/dat reader, `find_game()`, `game_version()`, `entry_fingerprint()` (hash of the compressed on-disk entry) |
| `exd` | EXH/EXD reader (big-endian) |
| `tex` | `.tex` header, mip layout and the one mip-dimension rule |
| `texdecode` | BC1–BC7 / uncompressed → RGBA (`texture2ddecoder`, Python ≤ 3.12), with numpy fallbacks for BC1–BC5 |
| `texwrite` | `.tex` writer; BC7 via `bc7enc` when texconv is not available |
| `bc7enc` | numpy BC7 encoder, modes 6 and 5 (slow; tests and the non-Windows fallback) |
| `mtrl`, `imcfile`, `mdlstrings`, `mdlpatch` | material / imc / model-string readers and in-place patchers |

`P:\projects\ffxiv-patchday\scripts-local` carries its own older `sqpack.py` and `exdfile.py`
so its snapshot tool stays stdlib-only. They have diverged from these; a fix to the format
readers here does not reach them automatically.

Python 3.11–3.12: `texture2ddecoder` publishes no 3.13 wheels.
