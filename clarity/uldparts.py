"""Sprite rectangles for `ui/uld/*.tex`, and a per-part upscale that does not bleed across them.

WHY THIS EXISTS. A UI .tex is an ATLAS: many unrelated sprites stitched into one sheet, and the
sheets are packed with NO GUTTER. JobHudXBM1 has 93 abutting pairs among 49 parts; JobHudXBM0 has
286 among 108. Concretely, parts 37 and 38 -- the Sunstrider and Moonstalker halves of the
Beastmaster wheel -- sit at x=250 and x=307, each 56 wide, so they share an edge, and that edge is
the exact pixel column the widget draws down the middle of the gauge.

Upscaling the sheet as one image gives any model with a receptive field wider than a pixel the
chance to mix each sprite into its neighbour across those seams. The artefact then lands on sprite
EDGES, which is where UI art is looked at hardest, and it is invisible in the sheet -- you only see
it once the game slices the rectangle back out.

WHAT THIS DOES. Every part is upscaled from its own crop, with reflect padding around it, and
composited back at exactly its rectangle times the tier scale. The scale is an integer, so
(u, v, w, h) maps to (Su, Sv, Sw, Sh) with no rounding and the repacked sheet is addressed correctly
by the .uld the game already ships -- which is essential, because the .uld is NOT modified and every
consumer reads those coordinates.

Reflect padding rather than zero or edge: it gives the model plausible signal outside the sprite
without inventing a hard border (zeros) or a smear (edge replication), and whatever it makes of that
padding is cropped away before compositing.

Pixels covered by no part -- the packing waste between sprites -- keep the whole-sheet upscale. The
game never samples them, so what happens there does not matter, but leaving them filled means the
output is still a sensible image to look at.

OVERLAPPING PARTS ARE NORMAL, AND ORDER RESOLVES THEM. Some sheets define a large rectangle with
smaller ones inside it. Compositing largest-first lets the small, specific part win the pixels it
shares, which is the one the game slices out at the tighter coordinates.

SHEETS ARE SHARED, SO NAME MATCHING ALONE IS NOT ENOUGH. `jobhudxbm0.uld` carries 108 parts across
FOUR sheets -- JobHudNumBg, JobHudSMN1, JobHudXBM0 and Parameter_Gauge -- of which only 16 are cut
from JobHudXBM0 itself. Parameter_Gauge.tex is described entirely by other widgets' ULDs, so looking
only at `parameter_gauge.uld` would leave the single most-shared sheet in the HUD on the whole-sheet
path. `build_index` therefore loads a whole set of ULDs and keys the rectangles by the sheet each
part actually names; `rects_for` falls back to the same-stem ULD when no index is supplied.

sqpack is a hash index with no directory listing, so the set of ULDs to load has to come from
somewhere -- the caller passes the stems it knows about, which for `clarity run` is every ui/uld row
in the manifest.
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# ULD coordinates are in BASE texture space. The shipped `_hr1` sheets are 2x, so every rectangle
# has to be doubled before it addresses one. Verified on JobHudXBM1_hr1 (KB 05).
HR1 = re.compile(r"_hr1$")

PAD = 8  # source pixels of reflect padding around each part

_MOD = []  # one-element cache for the uldfile module (or None)


def _uldfile():
    """`uldfile` lives with the rendering KB tools, not in ffxiv-kbtools beside sqpack/texfile.

    Imported lazily and tolerantly: clarity must still run for every other family on a machine where
    the rendering project is not mounted. A missing parser degrades ui-uld to whole-sheet upscaling,
    which is exactly the old behaviour, so it is a quality regression and never a failure.
    """
    if _MOD:
        return _MOD[0]
    mod = None
    try:
        import uldfile as mod  # ty: ignore[unresolved-import]
    except ImportError:
        import os
        import sys

        from . import paths

        cand = os.path.normpath(
            os.path.join(paths.PROJECTS, "ffxiv-rendering", "knowledge-base", "tools")
        )
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.append(cand)
            try:
                import uldfile as mod  # ty: ignore[unresolved-import]
            except ImportError:
                mod = None
    _MOD.append(mod)
    return mod


def uld_for(tex_path):
    """`ui/uld/jobhudxbm1_hr1.tex` -> (`ui/uld/jobhudxbm1.uld`, stem, 2)."""
    base = tex_path.rsplit("/", 1)[-1]
    stem_hr = base[:-4] if base.lower().endswith(".tex") else base
    stem = HR1.sub("", stem_hr)
    return f"ui/uld/{stem}.uld", stem, (2 if stem != stem_hr else 1)


def ulds_in_pathlist(file):
    """Every `ui/uld/*.uld` named by a ResLogger CurrentPathList.

    Guessing `ui/uld/<sheet>.uld` from the sheet names covers the widgets whose ULD is named after
    their own atlas, which is 369 of them -- but a shared sheet is described by ULDs with unrelated
    names, and there is no way to enumerate sqpack (it is a hash index, with no directory). The path
    list is the one place the real .uld names are written down, and `plan` already reads it for .tex.
    """
    if not file:
        return []
    opener = open
    if str(file).endswith(".gz"):
        import gzip

        opener = gzip.open
    out = set()
    try:
        with opener(file, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip().rsplit(",", 1)[-1].strip().lower()
                if p.startswith("ui/uld/") and p.endswith(".uld"):
                    out.add(p)
    except OSError:
        return []
    return sorted(out)


def build_index(gd, stems, log=None, pathlist=None):
    """{sheet stem (lowercase): [(u, v, w, h)] in BASE space} from a set of ULDs.

    `stems` is what the caller knows about -- for `clarity run`, the basenames of its ui/uld rows,
    each turned into a same-named .uld guess. `pathlist` adds the real .uld names when a ResLogger
    list is available. Each ULD is credited to the sheets its parts actually NAME, not to its own
    name, which is the whole point: that is how Parameter_Gauge gets its rectangles.
    """
    mod = _uldfile()
    idx = {}
    if mod is None:
        return idx
    loaded = 0
    cands = {"ui/uld/{}.uld".format(HR1.sub("", s.lower())) for s in stems}
    cands.update(ulds_in_pathlist(pathlist))
    for path in sorted(cands):
        try:
            if not gd.exists(path):
                continue
            u = mod.load(gd, path)
        except Exception as e:
            logger.debug("%s: could not parse (%s); skipped", path, e)
            continue
        loaded += 1
        by_asset = {aid: p.rsplit("/", 1)[-1].lower() for aid, p in u.assets}
        for plist in u.parts.values():
            for tex, x, y, w, h in plist:
                name = by_asset.get(tex)
                if not name or not name.endswith(".tex") or w <= 0 or h <= 0:
                    continue
                idx.setdefault(name[:-4], []).append((x, y, w, h))
    if log:
        log("uld index: %d .uld loaded, rectangles for %d sheet(s)" % (loaded, len(idx)))
    return idx


def rects_for(gd, tex_path, width=None, height=None, index=None, extra_ulds=()):
    """Sprite rectangles in the pixel space of THIS .tex, deduped and clipped. [] if unknown."""
    mod = _uldfile()
    if mod is None:
        return []
    up, stem, mult = uld_for(tex_path)
    stem_l = stem.lower()
    found_all = []
    if index is not None:
        found_all = [(0, 0, *r) for r in index.get(stem_l, ())]
    sources = () if index is not None else (up,)
    for path in sources + tuple(extra_ulds):
        try:
            if not gd.exists(path):
                continue
            found_all.extend(mod.load(gd, path).parts_for(stem))
        except Exception as e:
            logger.debug("%s: could not parse (%s); skipped", path, e)
            continue
    out, seen = [], set()
    for _lid, _i, x, y, w, h in found_all:
        if w <= 0 or h <= 0:
            continue
        r = (x * mult, y * mult, w * mult, h * mult)
        if width is not None and height is not None:
            # Clip to the sheet. A rectangle that runs off the edge is a bad rectangle, not a
            # licence to index out of bounds; one entirely outside is dropped.
            x0, y0 = min(r[0], width), min(r[1], height)
            x1, y1 = min(r[0] + r[2], width), min(r[1] + r[3], height)
            if x1 <= x0 or y1 <= y0:
                continue
            r = (x0, y0, x1 - x0, y1 - y0)
        # Dedupe: the same rectangle is routinely listed by several parts lists (and by several
        # ULDs, once shared sheets are in play). Upscaling it more than once costs time and cannot
        # change the answer.
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def abutting(rects):
    """How many pairs of rectangles touch or overlap -- the number this whole module is about."""
    n = 0
    for i in range(len(rects)):
        x0, y0, w0, h0 = rects[i]
        for j in range(i + 1, len(rects)):
            x1, y1, w1, h1 = rects[j]
            if x0 < x1 + w1 and x1 < x0 + w0 and y0 < y1 + h1 and y1 < y0 + h0:
                n += 1  # overlapping
            elif (x0 + w0 == x1 or x1 + w1 == x0) and y0 < y1 + h1 and y1 < y0 + h0:
                n += 1  # sharing a vertical edge
            elif (y0 + h0 == y1 or y1 + h1 == y0) and x0 < x1 + w1 and x1 < x0 + w0:
                n += 1  # sharing a horizontal edge
    return n


def _pad_crop(rgba, x, y, w, h, pad):
    """The part's crop, reflect-padded FROM ITSELF by `pad` pixels on every side.

    THE PADDING MUST NOT COME FROM THE ATLAS, and it used to. The first version took the pad from
    the sheet -- `rgba[y-py0:y+h+py1, x-px0:x+w+px1]` -- and reflected only the shortfall at the
    sheet's outer edge. For a part in the middle of a sheet there is no shortfall, so all 8 pixels
    of context were the NEIGHBOURING SPRITE: unrelated artwork, abutting with no gutter, which is
    the one thing this module exists to keep out. Cropping the model's output back to the rectangle
    afterwards does not undo that, because the neighbour is inside the receptive field of the
    output's edge pixels -- the exact pixels the docstring above calls the ones UI art is looked at
    hardest at. It reintroduced a weaker form of the whole-sheet artefact while reporting that it
    had prevented it.

    Reflect from the part's own pixels is what the module always claimed to do: plausible signal
    outside the sprite that is a continuation of the sprite, with no hard border and no neighbour.

    -> (crop, pad) where the returned pad is what was actually applied on each side, which can be
    less than asked for: numpy's `reflect` requires the pad to be SMALLER than the axis it reflects,
    and UI atlases contain 1- and 2-pixel parts (dividers, single-pixel gradient strips). Asking for
    8 around a 1px-tall part raises; clamping is not a nicety.
    """
    p = min(pad, max(0, w - 1), max(0, h - 1))
    crop = rgba[y : y + h, x : x + w]
    if p <= 0:
        return crop, 0
    return np.pad(crop, ((p, p), (p, p), (0, 0)), mode="reflect"), p


def upscale_by_parts(run_batch, rgba, scale, rects, whole=None, pad=PAD):
    """Upscale each rectangle from its own padded crop and composite at rect * scale.

    `run_batch(list_of_rgba_uint8) -> list_of_rgba_uint8 at `scale`` is the caller's upscale for a
    GROUP of same-shaped images, so this module stays ignorant of models and of roles.  `whole` is
    the already-computed whole-sheet upscale, used for the ground between sprites; None leaves it
    transparent.

    ONE CALL PER PART WAS THE WRONG SHAPE AND IT SHOWED UP AS 8x THE WORK. Measured across the
    queue: 5,370 ui/uld sheets carry 38,049 part rectangles, so a call per part is 43,419 model
    invocations where the old whole-sheet path made 5,370. Atlas parts are mostly TINY -- a divider,
    a corner, a 40x40 button face -- and on an image that size a model call is almost entirely
    launch overhead, which is the exact observation roles.do_ui_batch was written for. Parts are
    therefore grouped by padded shape and handed over a group at a time.

    Compositing still happens largest-first afterwards, independent of the batch order, so a small
    rectangle nested inside a big one still wins the pixels they share.

    -> uint8 (H*scale, W*scale, 4)
    """
    H, W = rgba.shape[:2]
    out = whole.copy() if whole is not None else np.zeros((H * scale, W * scale, 4), np.uint8)
    if scale <= 1 or not rects:
        return out
    s = scale
    groups = {}  # padded crop shape -> [(rect, pad, crop)]
    for r in rects:
        x, y, w, h = r
        crop, p = _pad_crop(rgba, x, y, w, h, pad)
        groups.setdefault(crop.shape, []).append((r, p, crop))
    done = {}
    for _shape, items in groups.items():
        for (r, p, _c), up in zip(items, run_batch([c for _r, _p, c in items]), strict=True):
            done[r] = (p, up)
    for x, y, w, h in sorted(rects, key=lambda r: -(r[2] * r[3])):
        p, up = done[(x, y, w, h)]
        if p:
            up = up[p * s : (p + h) * s, p * s : (p + w) * s]
        if up.shape[0] != h * s or up.shape[1] != w * s:
            # A role function that did not return exactly scale*input is a bug, but silently
            # writing a mis-sized block would corrupt the sheet at a coordinate the game trusts.
            # Leave the whole-sheet pixels for this part instead.
            continue
        out[y * s : (y + h) * s, x * s : (x + w) * s] = up
    return out
