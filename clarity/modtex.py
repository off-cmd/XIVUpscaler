"""Upscale the textures *inside an existing Penumbra mod* (the icon packs, G6): the mod folder is

mirrored to `<out>/<dir name> (upscaled)`, every JSON copied unchanged, every `.tex` replaced by
its upscaled twin at the same relative path. Duplicate files (same bytes — Color Gearset ships
each icon twice, Simpler Buff Icons reuses 8 designs across 96 ids) are processed once.

Roles come from the game path each file is mapped to (ui/icon → icon, ui/uld → ui, chara → the
usual). Icons ≤ 128 px go 4×, sheets 2×; uncompressed sources stay uncompressed, BC3 stays BC3,
other block formats become BC7.

KNOWN LIMIT: the duplicate cache is keyed on SOURCE BYTES ALONE, while processing also depends on
the role and family the game path classifies to and on the size-derived scale. Two byte-identical
files mapped to paths with different semantics -- the same art shipped as both an icon and a UI
sheet, say -- would reuse whichever was processed first. That cannot happen within a set of icon
packs, where every mapping is ui/icon, and fixing it properly means decoding before the cache
lookup rather than after, so it is recorded here rather than papered over.
"""

import hashlib
import logging
import os
import pathlib
import shutil

from . import manifest as mf
from . import texio
from .jsonio import read_json, write_json
from .processing import roles

logger = logging.getLogger(__name__)


def contained(base, rel):
    r"""Is `base/rel` still inside `base`? Mod JSON is somebody else's file, so this is checked.

    The Files map of a Penumbra mod is third-party data that arrives with the mod, and every value
    in it is used twice here: once to READ from the mod folder and once to WRITE the upscaled twin.
    Two spellings escape a plain os.path.join:

      * a ROOTED value ("D:\\x\\y.tex", "\\\\server\\share\\y.tex", or "/y.tex"). On Windows
        os.path.join throws away the base when the second part is rooted, so BOTH joins collapse to
        the same external path -- the read and the write become the same file, and the twin build
        would encode over somebody's original.
      * "..\\.." segments, which walk out of the mod and back down anywhere.

    Neither has been seen in the six icon packs -- no input was found containing one -- but
    "untrusted until read" is the rule for mod content on this machine, and the check costs a
    normpath.
    """
    b = os.path.normcase(os.path.normpath(os.path.abspath(base)))
    p = os.path.normcase(os.path.normpath(os.path.abspath(os.path.join(base, rel))))
    return p == b or p.startswith(b + os.sep)


def mod_files(mod_dir, log=None):
    """{relative file path: game path} over default_mod.json and every group option."""
    out = {}
    for name in sorted(os.listdir(mod_dir)):
        if not name.endswith(".json") or name == "meta.json":
            continue
        try:
            j = read_json(os.path.join(mod_dir, name))
        except (OSError, ValueError) as e:
            logger.debug("%s: not a readable JSON file (%s); skipped", name, e)
            continue
        blocks = [j] if "Files" in j else []
        for o in j.get("Options", []):
            blocks.append(o)
        for b in blocks:
            for gp, rel in b.get("Files", {}).items():
                rel = rel.replace("\\", "/")
                if not contained(mod_dir, rel):
                    if log:
                        log(f"  REFUSED (escapes the mod folder) {name} -> {rel}")
                    continue
                out[rel] = gp.lower()
    return out


def upscale_mod(
    mod_dir,
    out_root,
    engine,
    log=print,
    icon_scale=4,
    sheet_scale=2,
    suffix=" (upscaled)",
):
    name = os.path.basename(os.path.normpath(mod_dir))
    dst = os.path.join(out_root, name + suffix)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    files = mod_files(mod_dir, log=log)
    # the JSON may spell a path with different case than the disk (Windows does not care); key on
    # the on-disk spelling so the mirror has exactly one copy of each file
    ondisk = {}
    for root, _, fs in os.walk(mod_dir):
        rel_root = os.path.relpath(root, mod_dir)
        for f in fs:
            rel = os.path.normpath(os.path.join(rel_root, f)).replace("\\", "/")
            rel = rel[2:] if rel.startswith("./") else rel
            ondisk[rel.lower()] = rel
    files = {ondisk.get(rel.lower(), rel): gp for rel, gp in files.items()}
    # copy everything that is not a texture we will replace
    for rel in ondisk.values():
        src = os.path.join(mod_dir, rel)
        d = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if rel in files and rel.lower().endswith(".tex"):
            continue
        shutil.copy2(src, d)
    meta_p = os.path.join(dst, "meta.json")
    if os.path.isfile(meta_p):
        meta = read_json(meta_p)
        meta["Name"] = meta.get("Name", name) + suffix
        meta["Description"] = (
            "Upscaled twin built by clarity-upscale (icons %d×, sheets %d×). "
            % (icon_scale, sheet_scale)
        ) + meta.get("Description", "")
        write_json(meta_p, meta)
    cache, stats = {}, {"unique": 0, "dupes": 0, "skipped": 0, "written": 0}
    for rel, gp in sorted(files.items()):
        src = os.path.join(mod_dir, rel)
        if not os.path.isfile(src) or not rel.lower().endswith(".tex"):
            continue
        raw = pathlib.Path(src).read_bytes()
        key = hashlib.sha256(raw).hexdigest()
        d = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if key in cache:
            stats["dupes"] += 1
            pathlib.Path(d).write_bytes(cache[key]) if cache[key] else shutil.copy2(src, d)
            continue
        stats["unique"] += 1
        try:
            hdr, rgba = texio.read(raw)
            fam, _part, role = mf.classify(gp, hdr)
            if role in ("skip", "id", "other"):
                cache[key] = None
                stats["skipped"] += 1
                shutil.copy2(src, d)
                continue
            scale = icon_scale if max(hdr.width, hdr.height) <= 128 else sheet_scale
            top = {4: "4x", 2: "2x", 1: "native"}[scale]
            result = roles.process(engine, role, fam, rgba, hdr.format_name, top)
            img = result[top]
            fmt_out = texio.out_format(hdr.format_name, role)
            mips = texio.full_mips(img.shape[1], img.shape[0]) if hdr.mip_count > 1 else 1
            enc = texio.encode(img, fmt_out, mips=mips, attributes=hdr.attributes)
            pathlib.Path(d).write_bytes(enc)
            cache[key] = enc
            stats["written"] += 1
            log(
                "  %s %dx%d %s -> %dx%d (%s)"
                % (
                    rel,
                    hdr.width,
                    hdr.height,
                    hdr.format_name,
                    img.shape[1],
                    img.shape[0],
                    role,
                )
            )
        except Exception as e:
            log(f"  FAILED {rel}: {e}")
            shutil.copy2(src, d)
            cache[key] = None
    log(f"{name}: {stats}")
    return dst, stats
