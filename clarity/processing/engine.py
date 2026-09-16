"""Super-resolution engine: spandrel-loaded ESRGAN-family models on the GPU with tiling, or a

Lanczos fallback (CPU, no torch) so the plumbing runs anywhere.

Model slots → weight files. The tracked mapping is `clarity/models/registry.json` (package
data); `<models_dir>/registry.json` beside the weights overrides it per slot. The defaults are the
models Kartoffels' ChaiNNer chains use (KB 08), all on openmodeldb.info:

    bc1clean  1x_BC1-smooth2.pth            1×, run on every BC1 source before anything else
    normal    4x-Normal-RG0-BC7.pth         4×, tangent normals with B zeroed
    color     4x_scalenx_90k.pth            4×, general colour (diffuse / spec / base / world)
    face      4xFaceUpDAT.pth               4×, face base colour
    skin      x1_ITF_SkinDiffDDS_v1.pth     1×, skin de-artefact
    hair      4x_UltraFArt_v3.pth           4×, hair (unused by default: Hair Defined 2 covers hair)
    ui        4x_foolhardy_Remacri.pth      4×, UI sheets and icons (alpha handled outside the model)
"""

import os

import numpy as np

from .. import paths
from ..jsonio import read_json

DEFAULT_REGISTRY: dict[str, str | None] = {
    "bc1clean": "1x_BC1-smooth2.pth",  # set to null in registry.json when the colour model removes BC artefacts itself
    "normal": "4x-Normal-RG0-BC7.pth",  # BC7 / uncompressed normal sources
    "normal_bc1": "4x-Normal-RG0-BC1.pth",  # BC1 normal sources (trained on exactly that degradation); falls back to "normal"
    "color": "4x_scalenx_90k.pth",
    "mask": None,  # scalar channels; None = use "color"
    "face": "4xFaceUpDAT.pth",
    "skin": "x1_ITF_SkinDiffDDS_v1.pth",
    "hair": "4x_UltraFArt_v3.pth",
    "ui": "4x_foolhardy_Remacri.pth",
}
# The 2024–2025 picks (see KB 30-postprocess/17). Slot names match the tracked registry.
RECOMMENDED_REGISTRY = {
    "bc1clean": None,  # PBRify models were trained with BC/dds compression in the LR
    "normal": "4x-Normal-RG0-BC7.pth",
    "normal_bc1": "4x-Normal-RG0-BC1.pth",
    "color": "4x-PBRify_RPLKSRd_V3.pth",  # RealPLKSR-DySample, game textures, ~8x faster than DAT2; 4x-PBRify_UpscalerV4.pth for the slow, sharper DAT2
    "mask": "4x-PBRify_UpscalerSPANV4.pth",  # SPAN, fast: three passes per mask
    "face": "4xFaceUpDAT.pth",
    "skin": "x1_ITF_SkinDiffDDS_v1.pth",
    "hair": "4x_UltraFArt_v3.pth",
    "ui": "4x-UltraSharpV2.safetensors",  # DAT2, text/anti-aliasing aware; CC-BY-NC-SA (personal use)
}


class Engine:
    # tile defaults to 512 to match the CLI, so constructing an Engine directly behaves the
    # same as running the command. The two drifting apart is how a "default" stops meaning one.
    def __init__(
        self,
        models_dir,
        device=None,
        tile=512,
        pad=16,
        fp16=True,
        allow_fallback=True,
        tile_batch=1,
    ):
        self.models_dir = models_dir
        self.registry = dict(DEFAULT_REGISTRY)
        # registry.json beside the weights first, then the tracked one in scripts-local\clarity-upscale\models
        # (paths.REGISTRY) -- the tracked copy is the configuration of record and wins.
        for reg in ([os.path.join(models_dir, "registry.json")] if models_dir else []) + [
            paths.REGISTRY
        ]:
            if reg and os.path.isfile(reg):
                self.registry.update(read_json(reg))
        self.tile, self.pad, self.fp16 = tile, pad, fp16
        # Tiles of one source are independent, so they CAN go through as one batched forward
        # pass rather than one call each; this is the cap on how many. It defaults to 1 -- off.
        # A 1024x1024 through the DAT-based colour model at 4 tiles a pass wanted ~32 GB on a
        # 12 GB card, and on Windows the NVIDIA driver's sysmem fallback served it from system
        # RAM over PCIe instead of raising: 253 s for the texture, no exception, so the
        # OOM-halving below never ran. Batching pays only for light archs with headroom; opt in
        # with --tile-batch. When it is on, the cap is lowered in place on the first CUDA OOM and
        # stays lowered, so a run that guesses too high pays for it once.
        self.tile_batch = max(1, int(tile_batch))
        self.allow_fallback = allow_fallback
        self._models = {}
        self.torch = None
        self.device = "cpu"
        self.missing = set()
        try:
            import torch

            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass

    # ------------------------------------------------------------------ models
    def path(self, slot):
        if not self.models_dir or not self.registry.get(slot):
            return None
        p = os.path.join(self.models_dir, str(self.registry[slot]))
        return p if os.path.isfile(p) else None

    def has(self, slot: str) -> bool:
        return self.torch is not None and self.path(slot) is not None

    def enabled(self, slot: str) -> bool:
        """A slot mapped to null in registry.json is deliberately off (no fallback)."""
        return bool(self.registry.get(slot))

    def resolve(self, slot):
        """Slot aliases: normal_bc1 → normal, mask → color when unset."""
        if slot == "normal_bc1" and not self.has("normal_bc1"):
            return "normal"
        if slot == "mask" and not self.has("mask"):
            return "color"
        return slot

    def describe(self):
        return ", ".join(
            f"{s}={os.path.basename(self.path(s))}" for s in self.registry if self.has(s)
        )

    def model(self, slot):
        if slot in self._models:
            return self._models[slot]
        from spandrel import ModelLoader

        m = ModelLoader().load_from_file(self.path(slot))
        m = m.to(self.device).eval()
        if self.fp16 and self.device == "cuda" and m.supports_half:
            m = m.half()
        self._models[slot] = m
        return m

    def evict_others(self, keep):
        """Drop every cached model except `keep` and return whether anything was dropped.

        Models load lazily and are never otherwise released, and one run touches most of the
        slots (a colour texture alone reaches bc1clean, color and ui). On a card where they all
        fit that costs nothing; on one where they do not, the resident ones are the only thing
        left to give back when a single tile fails to allocate. This is that: called from the
        OOM path once tile batching is already at 1, before the tile is declared a failure.
        """
        dropped = [s for s, m in self._models.items() if m is not keep]
        for s in dropped:
            del self._models[s]
        if dropped and self.device == "cuda":
            self._torch().cuda.empty_cache()
        return bool(dropped)

    # ------------------------------------------------------------------ inference
    def run(self, slot, img, scale):
        """img: float32 (H, W, C) in [0, 1] → float32 (H*scale, W*scale, C). If the slot's model is

        missing (or torch is), falls back to Lanczos and records the slot in `self.missing`.
        """
        slot = self.resolve(slot)
        if self.has(slot):
            return self._run_model(slot, img, scale)
        if not self.allow_fallback:
            raise RuntimeError(f"model for slot {slot!r} not available ({self.registry.get(slot)})")
        self.missing.add(slot)
        return lanczos(img, scale)

    def run_batch(self, slot, imgs, scale):
        """Several same-sized images through one forward pass (the batch dimension never mixes, so

        each result equals `run` on that image alone). Falls back per image like `run`.
        """
        if not imgs:
            return []
        slot = self.resolve(slot)
        if self.has(slot) and len({i.shape for i in imgs}) == 1:
            # How many fit in one forward pass. The cost is batch x the area actually pushed
            # through, which is the tile for anything larger than a tile and the image itself for
            # anything smaller -- and the icon families are all smaller (40x40, 80x80), which is why
            # keying this off the tile alone was wrong: at `--tile 512` it computed 1 and quietly
            # disabled batching for exactly the textures that need it most.
            ih, iw = imgs[0].shape[:2]
            area = min(ih, self.tile) * min(iw, self.tile)
            per = max(1, (self.tile * self.tile) // max(1, area))
            out = []
            for s in range(0, len(imgs), per):
                chunk = imgs[s : s + per]
                try:
                    out.extend(
                        self._run_model(slot, chunk, scale)
                        if len(chunk) > 1
                        else [self._run_model(slot, chunk[0], scale)]
                    )
                except RuntimeError as e:  # CUDA OOM on a batch: fall back to one at a time
                    if _not_oom(e) or len(chunk) == 1:
                        raise
                    self._torch().cuda.empty_cache()
                    out.extend(self._run_model(slot, i, scale) for i in chunk)
            return out
        return [self.run(slot, i, scale) for i in imgs]

    def _torch(self):
        """torch, for the paths that only run when a model was loaded."""
        if self.torch is None:  # pragma: no cover - guarded by has()/model() before every call
            raise RuntimeError("torch is not available; Engine.run() falls back to Lanczos")
        return self.torch

    def _run_model(self, slot, img, scale):
        torch = self._torch()
        m = self.model(slot)
        ms = m.scale
        batch = isinstance(img, (list, tuple))
        imgs = list(img) if batch else [img]
        _H, _W, C = imgs[0].shape
        cin = m.input_channels
        xs = []
        for x in imgs:
            if cin == 1 and C != 1:
                x = x.mean(2, keepdims=True)
            elif cin == 3 and C == 1:
                x = np.repeat(x, 3, 2)
            elif cin == 3 and C == 4:
                x = x[..., :3]
            xs.append(np.ascontiguousarray(x.transpose(2, 0, 1)))
        t = torch.from_numpy(np.stack(xs)).to(self.device)
        if self.fp16 and self.device == "cuda" and m.supports_half:
            t = t.half()
        out = self._tiled(m, t, ms)
        ys = []
        for n in range(len(imgs)):
            y = out[n].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            if y.shape[2] != C:
                if C == 1:
                    y = y.mean(2, keepdims=True)
                else:
                    y = np.repeat(y[..., :1], C, 2) if y.shape[2] == 1 else y[..., :C]
            if ms != scale:
                y = lanczos(y, scale / ms)
            ys.append(y.astype(np.float32))
        return ys if batch else ys[0]

    def _tiled(self, m, t, ms):
        """One forward pass per tile group, writing each tile into its place in the output.

        Tiles are grouped by the shape actually extracted -- the padded window is narrower at the
        image edges -- and each group goes through the model as one batch. Padding the edge tiles
        up to a common size instead would feed the model invented border content and change the
        result; grouping leaves every tile's input tensor exactly what it is tile-by-tile, and the
        batch dimension never mixes, so the output is bitwise what the sequential loop produced.
        """
        torch = self._torch()
        N, _C, H, W = t.shape
        tile, pad = self.tile, self.pad
        if tile + 2 * pad >= H and tile + 2 * pad >= W:
            evicted = False
            while True:
                try:
                    with torch.no_grad():
                        return m(_pad8(t))[:, :, : H * ms, : W * ms]
                except RuntimeError as e:
                    if _not_oom(e) or evicted or not self.evict_others(m):
                        raise
                    evicted = True
        out = torch.zeros((N, m.output_channels, H * ms, W * ms), dtype=t.dtype, device=t.device)
        groups: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
                py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                py1, px1 = min(H, y1 + pad), min(W, x1 + pad)
                groups.setdefault((py1 - py0, px1 - px0), []).append((y0, x0, y1, x1, py0, px0))
        evicted = False
        for (dh, dw), windows in groups.items():
            s = 0
            while s < len(windows):
                chunk = windows[s : s + max(1, self.tile_batch // N)]
                try:
                    self._tile_chunk(m, t, ms, out, chunk, dh, dw, n_img=N)
                except RuntimeError as e:
                    if _not_oom(e):
                        raise
                    torch.cuda.empty_cache()
                    if len(chunk) > 1:  # a batch did not fit: halve the cap, keep it halved
                        self.tile_batch = max(1, (len(chunk) * N) // 2)
                        continue
                    # ONE tile did not fit. The other slots' resident models are the last
                    # thing to give back; drop them and try the tile once more. If it still
                    # does not fit, it really does not fit, and the texture fails with a
                    # message that says so rather than a run that crawls.
                    if evicted or not self.evict_others(m):
                        raise
                    evicted = True
                    continue
                s += len(chunk)
        return out

    def _tile_chunk(self, m, t, ms, out, chunk, dh, dw, n_img):
        """Run one group of equally shaped tiles and scatter the results into `out`."""
        torch = self._torch()
        views = [t[:, :, py0 : py0 + dh, px0 : px0 + dw] for _, _, _, _, py0, px0 in chunk]
        sub = views[0] if len(views) == 1 else torch.cat(views, 0)
        with torch.no_grad():
            o = m(_pad8(sub))
        for j, (y0, x0, y1, x1, py0, px0) in enumerate(chunk):
            oy, ox = (y0 - py0) * ms, (x0 - px0) * ms
            out[:, :, y0 * ms : y1 * ms, x0 * ms : x1 * ms] = o[
                j * n_img : (j + 1) * n_img, :, oy : oy + (y1 - y0) * ms, ox : ox + (x1 - x0) * ms
            ]


def _not_oom(e):
    """True unless `e` is torch's CUDA out-of-memory RuntimeError (the retry paths key on it)."""
    return "out of memory" not in str(e).lower()


def _pad8(t):
    """Pad to a multiple of 8 (many arches need it); the caller crops the result.

    Reflection has to read a row for every row it invents, so torch refuses a pad that is not
    strictly smaller than the dimension: a 4x4 texture wants 4 and gets
    "padding (0, 4) at dimension 3 of input [1, 3, 4, 4]". The game has such textures --
    `e8805/v01_c0101e8805_glv_n` is 4x4 -- so the degenerate case falls back to replicate, which
    has no such rule. It differs from reflect only in the invented rows, and those are exactly
    the ones cropped off the result.
    """
    import torch.nn.functional as F  # noqa: N812 - the torch convention

    _, _, h, w = t.shape
    ph, pw = (-h) % 8, (-w) % 8
    if not (ph or pw):
        return t
    mode = "reflect" if (pw < w and ph < h) else "replicate"
    return F.pad(t, (0, pw, 0, ph), mode=mode)


def lanczos(img, scale):
    """PIL Lanczos per channel; float32 (H, W, C) → float32."""
    from PIL import Image

    H, W, C = img.shape
    nh, nw = max(1, round(H * scale)), max(1, round(W * scale))
    out = np.empty((nh, nw, C), np.float32)
    for c in range(C):
        # Resample in float32 ("F") rather than through a 16-bit integer image: Pillow 13
        # removes the `mode=` argument fromarray() needed for "I;16", and the float path
        # loses nothing, since the input is float32 already.
        im = Image.fromarray(np.clip(img[..., c], 0, 1).astype(np.float32))
        out[..., c] = np.asarray(im.resize((nw, nh), Image.Resampling.LANCZOS), np.float32)
    return np.clip(out, 0, 1)


def box_down(img, factor):
    """Area-average downsample by an integer factor (used to derive the lower tiers)."""
    H, W, C = img.shape
    h, w = H // factor, W // factor
    return (
        img[: h * factor, : w * factor]
        .reshape(h, factor, w, factor, C)
        .mean((1, 3))
        .astype(np.float32)
    )
