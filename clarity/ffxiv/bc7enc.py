"""A BC7 encoder in numpy: mode 6 (one subset, RGBA 7-bit endpoints + p-bit, 4-bit indices),

with mode 5 (RGB line + separate alpha line) tried per block for the blocks mode 6 fits badly.

Why write one: Pillow only encodes DXT1/3/5 and BC5, and its DXT5 on a 2048² hair normal
measured 29.7 dB on the R channel (max error 90 of 255) -- strand normals need more than four
levels per block. No BC7 encoder on PyPI runs on Linux (texfury ships a Windows DLL). Mode 6
gives 16 interpolation levels between two 8-bit RGBA endpoints, which is what BC7 encoders
spend most of their time in anyway; mode 5 covers the blocks where alpha and colour do not
share a line. Every block is checked against `texture2ddecoder.decode_bc7` in `roundtrip()`.

Block layout, bits packed LSB-first (Khronos data format spec, §BC7):

    mode 6:  [mode 0b1000000 (7)] R0 R1 G0 G1 B0 B1 A0 A1 (7 each) P0 P1 (1 each)
             idx0 (3) idx1..15 (4 each)                                      = 128 bits
    mode 5:  [mode 0b100000 (6)] rot (2) R0 R1 G0 G1 B0 B1 (7 each) A0 A1 (8 each)
             colour idx0 (1) idx1..15 (2 each)  alpha idx0 (1) idx1..15 (2 each) = 128 bits

Interpolation: c = ((64 - w) * e0 + w * e1 + 32) >> 6 with w from the 2- or 4-bit weight tables.
"""

import numpy as np

W4 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], np.int64)
W2 = np.array([0, 21, 43, 64], np.int64)


# ------------------------------------------------------------------ helpers
def _blocks(rgba):
    """(h, w, 4) uint8 -> (B, 16, 4) float64 in raster block order, padded to multiples of 4."""
    h, w = rgba.shape[:2]
    H, Wd = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    a = np.zeros((H, Wd, 4), np.uint8)
    a[:h, :w] = rgba
    if h < H:
        a[h:] = a[h - 1 : h]
    if Wd > w:
        a[:, w:] = a[:, w - 1 : w]
    b = a.reshape(H // 4, 4, Wd // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4)
    return b.astype(np.float64), H // 4, Wd // 4


def _fit_line(X):
    """Principal direction of each block's texels: (B, C).

    The largest eigenvector of each block's 4x4 (or 3x3) covariance, from ``eigh``. The
    earlier power iteration started from (1, 1, 1, 1) and could never leave it when the true
    axis was orthogonal to that start -- a block with R rising while B falls by the same
    amount, say -- so both endpoints collapsed to the block mean and the block decoded flat
    with errors of 30 counts. A flat block never looks wrong at a glance, which is how the
    encoder shipped with it.
    """
    mu = X.mean(1, keepdims=True)
    Y = X - mu
    C = np.einsum("bic,bid->bcd", Y, Y)
    _w, v = np.linalg.eigh(C)  # ascending eigenvalues; the principal axis is the last column
    d = v[:, :, -1]
    return mu[:, 0], d


def _assign(X, e0, e1, weights):
    """Nearest interpolation level per texel. X (B,16,C), e0/e1 (B,C) -> idx (B,16)."""
    w = weights[None, :, None] / 64.0
    pal = (1 - w) * e0[:, None, :] + w * e1[:, None, :]  # (B, L, C)
    d = ((X[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)  # (B, 16, L)
    return d.argmin(-1)


def _lsq(X, idx, weights):
    """Least-squares endpoints for fixed indices."""
    w = weights[idx] / 64.0  # (B, 16)
    a = 1 - w
    A00 = (a * a).sum(1)
    A01 = (a * w).sum(1)
    A11 = (w * w).sum(1)
    b0 = np.einsum("bi,bic->bc", a, X)
    b1 = np.einsum("bi,bic->bc", w, X)
    det = A00 * A11 - A01 * A01
    ok = np.abs(det) > 1e-9
    e0 = np.empty_like(b0)
    e1 = np.empty_like(b1)
    e0[ok] = (A11[ok, None] * b0[ok] - A01[ok, None] * b1[ok]) / det[ok, None]
    e1[ok] = (A00[ok, None] * b1[ok] - A01[ok, None] * b0[ok]) / det[ok, None]
    m = X.mean(1)
    e0[~ok] = m[~ok]
    e1[~ok] = m[~ok]
    return np.clip(e0, 0, 255), np.clip(e1, 0, 255)


def _refine(X, weights, iters=4):
    mu, d = _fit_line(X)
    t = np.einsum("bic,bc->bi", X - mu[:, None], d)
    e0 = mu + t.min(1)[:, None] * d
    e1 = mu + t.max(1)[:, None] * d
    e0, e1 = np.clip(e0, 0, 255), np.clip(e1, 0, 255)
    for _ in range(iters):
        idx = _assign(X, e0, e1, weights)
        e0, e1 = _lsq(X, idx, weights)
    return e0, e1


def _err(X, e0, e1, idx, weights):
    w = weights[idx][:, :, None] / 64.0
    rec = np.floor(((64 - w * 64) * e0[:, None] + (w * 64) * e1[:, None] + 32) / 64)
    return ((X - rec) ** 2).sum((1, 2)), rec


# ------------------------------------------------------------------ mode 6
def _mode6(X):
    """-> (err (B,), lo (B,) uint64, hi (B,) uint64)."""
    B = X.shape[0]
    e0, e1 = _refine(X, W4)
    best = None
    for p0 in (0, 1):
        for p1 in (0, 1):
            q0 = np.clip(np.round((e0 - p0) / 2), 0, 127)
            q1 = np.clip(np.round((e1 - p1) / 2), 0, 127)
            r0, r1 = q0 * 2 + p0, q1 * 2 + p1
            idx = _assign(X, r0, r1, W4)
            # anchor: texel 0's index must have its top bit clear
            swap = idx[:, 0] >= 8
            r0s = np.where(swap[:, None], r1, r0)
            r1s = np.where(swap[:, None], r0, r1)
            q0s = np.where(swap[:, None], q1, q0)
            q1s = np.where(swap[:, None], q0, q1)
            p0s = np.where(swap, p1, p0)
            p1s = np.where(swap, p0, p1)
            idx = np.where(swap[:, None], 15 - idx, idx)
            err, _ = _err(X, r0s, r1s, idx, W4)
            cand = (
                err,
                q0s.astype(np.uint64),
                q1s.astype(np.uint64),
                p0s.astype(np.uint64),
                p1s.astype(np.uint64),
                idx.astype(np.uint64),
            )
            if best is None:
                best = cand
            else:
                better = err < best[0]
                best = tuple(
                    np.where(better[:, None] if n.ndim == 2 else better, n, o)
                    for n, o in zip(cand, best, strict=True)
                )
    err, q0, q1, p0, p1, idx = best
    lo = np.full(B, 1 << 6, np.uint64)  # mode 6
    pos = 7
    for c in range(4):
        lo |= q0[:, c] << np.uint64(pos)
        pos += 7
        lo |= q1[:, c] << np.uint64(pos)
        pos += 7
    lo |= p0 << np.uint64(63)
    hi = p1.copy()
    hi |= idx[:, 0] << np.uint64(1)
    for i in range(1, 16):
        hi |= idx[:, i] << np.uint64(4 + 4 * (i - 1))
    return err, lo, hi


# ------------------------------------------------------------------ mode 5
def _mode5(X):
    B = X.shape[0]
    Xc, Xa = X[:, :, :3], X[:, :, 3:4]
    c0, c1 = _refine(Xc, W2)
    a0, a1 = _refine(Xa, W2)
    qc0 = np.clip(np.round(c0 / 2), 0, 127)
    qc1 = np.clip(np.round(c1 / 2), 0, 127)
    rc0, rc1 = qc0 * 2, qc1 * 2  # mode 5 colour: 7 bits, expanded <<1
    qa0 = np.clip(np.round(a0), 0, 255)
    qa1 = np.clip(np.round(a1), 0, 255)
    ic = _assign(Xc, rc0, rc1, W2)
    ia = _assign(Xa, qa0, qa1, W2)
    swc = ic[:, 0] >= 2
    rc0s = np.where(swc[:, None], rc1, rc0)
    rc1s = np.where(swc[:, None], rc0, rc1)
    qc0s = np.where(swc[:, None], qc1, qc0)
    qc1s = np.where(swc[:, None], qc0, qc1)
    ic = np.where(swc[:, None], 3 - ic, ic)
    swa = ia[:, 0] >= 2
    qa0s = np.where(swa[:, None], qa1, qa0)
    qa1s = np.where(swa[:, None], qa0, qa1)
    ia = np.where(swa[:, None], 3 - ia, ia)
    ec, _ = _err(Xc, rc0s, rc1s, ic, W2)
    ea, _ = _err(Xa, qa0s, qa1s, ia, W2)
    err = ec + ea
    qc0s, qc1s, qa0s, qa1s = (v.astype(np.uint64) for v in (qc0s, qc1s, qa0s, qa1s))
    ic, ia = ic.astype(np.uint64), ia.astype(np.uint64)
    lo = np.full(B, 1 << 5, np.uint64)  # mode 5, rotation 0 (bits 6-7)
    pos = 8
    for c in range(3):
        lo |= qc0s[:, c] << np.uint64(pos)
        pos += 7
        lo |= qc1s[:, c] << np.uint64(pos)
        pos += 7
    # pos == 50: A0 8 bits (50..57), A1 8 bits (58..65) straddles the word boundary
    lo |= qa0s[:, 0] << np.uint64(50)
    lo |= (qa1s[:, 0] & np.uint64(0x3F)) << np.uint64(58)
    hi = (qa1s[:, 0] >> np.uint64(6)) & np.uint64(0x3)
    # colour indices: bit 66 = hi bit 2: idx0 (1 bit), then 2 bits each -> 31 bits (hi 2..32)
    hi |= ic[:, 0] << np.uint64(2)
    for i in range(1, 16):
        hi |= ic[:, i] << np.uint64(3 + 2 * (i - 1))
    # alpha indices: bit 97 = hi bit 33: idx0 (1 bit), then 2 bits each -> 31 bits (hi 33..63)
    hi |= ia[:, 0] << np.uint64(33)
    for i in range(1, 16):
        hi |= ia[:, i] << np.uint64(34 + 2 * (i - 1))
    return err, lo, hi


# ------------------------------------------------------------------ public
def encode(rgba, modes=(6, 5), chunk=16384):
    """(h, w, 4) uint8 RGBA -> BC7 block bytes (row-major blocks). Chunked: the assignment

    step is (blocks, 16, 16, 4) doubles, 4 MB per thousand blocks.
    """
    X, _bh, _bw = _blocks(np.asarray(rgba, np.uint8))
    out = np.empty((X.shape[0], 2), "<u8")
    for s in range(0, X.shape[0], chunk):
        Xc = X[s : s + chunk]
        best = None
        for m in modes:
            err, lo, hi = _mode6(Xc) if m == 6 else _mode5(Xc)
            if best is None:
                best = [err, lo, hi]
            else:
                better = err < best[0]
                best = [
                    np.where(better, err, best[0]),
                    np.where(better, lo, best[1]),
                    np.where(better, hi, best[2]),
                ]
        assert best is not None
        out[s : s + chunk, 0] = best[1]
        out[s : s + chunk, 1] = best[2]
    return out.tobytes()


def roundtrip(rgba, blocks=None):
    """PSNR per channel of the encoded image, decoded by texture2ddecoder. -> (psnr[4], mean|err|[4])"""
    import texture2ddecoder as td

    rgba = np.asarray(rgba, np.uint8)
    h, w = rgba.shape[:2]
    if blocks is None:
        blocks = encode(rgba)
    dec = np.frombuffer(td.decode_bc7(blocks, w, h), np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3]]
    err = np.abs(dec.astype(float) - rgba.astype(float))
    mse = (err**2).mean((0, 1))
    return 10 * np.log10(255**2 / np.maximum(mse, 1e-9)), err.mean((0, 1))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    img = (rng.random((64, 64, 4)) * 255).astype(np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    img[..., 0] = (xx * 4) & 255
    img[..., 1] = (yy * 4) & 255
    img[..., 2] = 200
    img[..., 3] = ((xx + yy) * 2) & 255
    for modes in ((6,), (5,), (6, 5)):
        p, e = roundtrip(img, encode(img, modes))
        print("modes", modes, "PSNR", p.round(1), "mean|err|", e.round(2))  # noqa: T201 - self-test entry point
