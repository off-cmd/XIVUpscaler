"""`uv run clarity <command>` — see README.md for the runbook."""

import argparse
import logging
import os
import pathlib
import sys
import time
import traceback

import numpy as np

from . import ffxiv as kb
from . import manifest as mf
from . import paths, texio, uldparts
from .packaging import penumbra as packer
from .packaging import release
from .processing import roles
from .processing.engine import Engine


def cmd_plan(a):
    man = mf.Manifest(a.db)
    gd = kb.game()
    if a.chara:
        mf.gen_chara(man, gd)
    if a.icons:
        mf.gen_icons(man, gd)
    if a.pathlist:
        mf.from_pathlist(man, gd, a.pathlist)
    # Rows inserted by THIS plan already carry their verdict (Manifest.add consults skip_reason), so
    # this only ever catches rows from an older enumeration or an older classifier. Run it here
    # rather than leaving it to the operator: `plan` is the only command that creates rows, so it is
    # the only place the queue can go stale, and a queue depth that counts work nobody will ever do
    # is worse than useless -- it hides the real number.
    n, why = mf.sync_skipped(man)
    if n:
        print("status sync: %d planned row(s) had no processing path and are now 'skipped'" % n)
        for k, c_ in why.most_common():
            print("    %-52s %6d" % (k[:52], c_))
    cmd_estimate(a)


def cmd_estimate(a):
    man = mf.Manifest(a.db)
    print(
        "%-14s %-7s %-8s %7s %10s %10s %10s %10s"
        % ("family", "role", "status", "n", "src MB", "native MB", "2x MB", "4x MB")
    )
    tot: dict[str, float] = {"src": 0.0, "native": 0.0, "2x": 0.0, "4x": 0.0}
    for family, role, status, n, b in man.summary():
        if role in ("id", "skip", "other"):
            print(
                "%-14s %-7s %-8s %7d %10.0f %10s %10s %10s"
                % (family, role, status, n, (b or 0) / 1e6, "-", "-", "-")
            )
            continue
        # BC7 at 8 bpp (2× BC1) with mips ≈ ×1.33
        rows = man.db.execute(
            "SELECT w,h,fmt FROM tex WHERE family=? AND role=? AND status=?",
            (family, role, status),
        ).fetchall()
        est = {"native": 0.0, "2x": 0.0, "4x": 0.0}
        for w, h, fmt in rows:
            top = roles.top_tier(family, w, h, a.top)
            if top is None:
                continue
            bpp = 4 if fmt in ("B8G8R8A8", "B8G8R8X8") else 1
            for tier in roles.tiers_below(top):
                s = roles.TIER_SCALE[tier]
                est[tier] += w * h * s * s * bpp * 1.33
        print(
            "%-14s %-7s %-8s %7d %10.0f %10.0f %10.0f %10.0f"
            % (
                family,
                role,
                status,
                n,
                (b or 0) / 1e6,
                est["native"] / 1e6,
                est["2x"] / 1e6,
                est["4x"] / 1e6,
            )
        )
        tot["src"] += b or 0
        for k, v in est.items():
            tot[k] += v
    print(
        "TOTAL src {:.1f} GB -> native {:.1f} GB, 2x {:.1f} GB, 4x {:.1f} GB (each tier is a full separate set)".format(
            tot["src"] / 1e9, tot["native"] / 1e9, tot["2x"] / 1e9, tot["4x"] / 1e9
        )
    )


def check_sysmem_fallback(engine):
    """Say so when the NVIDIA driver will serve CUDA from system RAM instead of raising.

    Windows drivers since 2023 default "CUDA - Sysmem Fallback Policy" to spilling: an
    allocation the card cannot hold succeeds, backed by system RAM over PCIe, instead of
    failing with out-of-memory. Every OOM-recovery path in this program (Engine halving its
    tile batch, evicting models, failing a texture with a reason) keys on that exception. With
    the fallback on none of them ever runs -- the run does not fail, it crawls: 253 s for a
    texture that takes 18 s, and a nine-hour run at 2.4 s/texture that Task Manager showed
    holding 35 GB. The probe asks for more than the card has; if that is granted, the policy
    is on. It costs one allocation and is freed before this returns.
    """
    torch = engine.torch
    if torch is None or engine.device != "cuda":
        return
    try:
        _free, total = torch.cuda.mem_get_info()
        probe = torch.empty(total + (256 << 20), dtype=torch.uint8, device="cuda")
    except Exception:
        return  # refused, as it should be: allocations that do not fit will raise
    del probe
    torch.cuda.empty_cache()
    print(
        "WARNING: CUDA allocations larger than VRAM are being served from system RAM"
        " (NVIDIA 'CUDA - Sysmem Fallback Policy'). A texture that does not fit will not fail,"
        " it will run over PCIe at a fraction of the speed and the run will never say why."
        " Set it to 'Prefer No Sysmem Fallback' in NVIDIA Control Panel > Manage 3D Settings."
    )


def check_encoder(a):  # noqa: ARG001 - same shape as every cmd_* handler
    """Refuse to burn days on a CPU BC7 codec: probe texconv once and report what it uses."""
    if not texio.use_texconv():
        print("encoder: numpy bc7enc (slow; fine for tests only)")
        return True
    p = texio.probe_texconv()
    if p.get("debug"):
        print(
            f"WARNING: {texio.TEXCONV} is a Debug build (imports ucrtbased.dll) — it cannot create a D3D device and uses an unoptimised CPU codec."
        )
    if p["ok"] and p["gpu"]:
        line = [ln for ln in p["text"].splitlines() if "DirectCompute" in ln]
        print(
            "encoder: texconv on the GPU — {} ({:.2f}s for a 64x64 probe)".format(
                line[0].strip() if line else "?", p["seconds"]
            )
        )
        return True
    print("encoder: texconv says: %s" % (p["text"].splitlines()[-1] if p["text"] else "no output"))
    if os.environ.get("CLARITY_ALLOW_CPU_BC7") == "1":
        print("CLARITY_ALLOW_CPU_BC7=1 — continuing on the CPU codec anyway")
        return True
    print(
        "STOP: texconv is not using DirectCompute (this is the all-cores-pegged, GPU-idle pattern: ~1 min per tier).\n"
        "      Put a release texconv.exe (microsoft/DirectXTex releases) in tools\\ or point CLARITY_TEXCONV at one,\n"
        "      or set CLARITY_ALLOW_CPU_BC7=1 to accept the cost."
    )
    return False


def estimate_output_bytes(man, families, top_override=None, since=None, status="planned"):
    """Bytes the queued rows will WRITE, per tier. Same arithmetic as `estimate`, restricted to the

    families this run will actually walk.
    """
    tot: dict[str, float] = {"native": 0.0, "2x": 0.0, "4x": 0.0}
    n = 0
    q = "SELECT family,w,h,fmt FROM tex WHERE status=?"
    args = [status]
    if families:
        q += " AND family IN ({})".format(",".join("?" * len(families)))
        args += list(families)
    if since:
        # The same patch-delta clause Manifest.rows(since=) applies, so the estimate counts
        # the rows a `--since` run will walk rather than everything still planned.
        q += " AND (updated >= ? OR note LIKE 'changed at %')"
        args.append(since)
    for fam, w, h, fmt in man.db.execute(q, args):
        top = roles.top_tier(fam, w, h, top_override)
        if top is None:
            continue
        n += 1
        bpp = 4 if fmt in ("B8G8R8A8", "B8G8R8X8") else 1
        for tier in roles.tiers_below(top):
            s = roles.TIER_SCALE[tier]
            tot[tier] += w * h * s * s * bpp * 1.33
    return n, tot


def check_disk(a, man, families):
    """Refuse to start a multi-day run that cannot fit its own output.

    THIS IS THE FAILURE THAT COSTS THE MOST AND WARNS THE LEAST. Every other way this pipeline can
    go wrong announces itself: a bad codec is slow from the first texture, a missing model is named
    at startup, a decode failure is one row. Running out of disk happens at hour 180 of 200, after
    the electricity is already spent, and what it leaves behind is a half-written texture and a
    manifest that disagrees with the mod folder.

    Measured on this install 2026-09-10: 404 GB still to write, 363 GB free on the volume holding
    the Penumbra root. The queue does not fit, and nothing anywhere said so -- `check_encoder`
    guards the encoder and there was no equivalent for the disk.

    Advisory by default rather than fatal, because "free space" is a moving number on a machine
    someone is also using; --strict-disk makes it a stop.
    """
    import shutil

    try:
        usage = shutil.disk_usage(a.out if os.path.isdir(a.out) else os.path.dirname(a.out) or ".")
    except OSError as e:
        print(f"disk: could not measure free space at {a.out} ({e})")
        return True
    n, tot = estimate_output_bytes(man, families, a.top, since=getattr(a, "since", None))
    want = sum(tot[t] for t in tot if not a.tiers or t in a.tiers)
    free = usage.free
    print(
        "disk: %s has %.0f GB free; this run's %d queued row(s) will write about %.0f GB"
        % (a.out, free / 1e9, n, want / 1e9)
    )
    for t in ("4x", "2x", "native"):
        if tot[t] and (not a.tiers or t in a.tiers):
            print("        %-7s %6.0f GB" % (t, tot[t] / 1e9))
    if want <= free * 0.92:
        return True
    print("WARNING: this does not fit. Short by about %.0f GB." % ((want - free) / 1e9))
    print("         Options, cheapest first:")
    print("           --tiers 2x            one tier instead of three")
    print("           --family <name>       the families you actually look at")
    print("           --out <path>          a volume with room (CLARITY_OUT also sets it)")
    print("         Each tier is a COMPLETE separate set, so three tiers is three times the bytes.")
    if a.strict_disk:
        print("STOP: --strict-disk was given.")
        return False
    print("         Continuing anyway (pass --strict-disk to make this a stop).")
    return True


def cmd_run(a):
    if a.profile == "everyday" and a.tiers:
        raise ValueError(
            "--tiers requires --profile legacy; everyday selects one tier per resource"
        )
    man = mf.Manifest(a.db)
    engine = Engine(
        a.models,
        device=a.device,
        tile=a.tile,
        tile_batch=a.tile_batch,
        allow_fallback=not a.strict,
    )
    print(
        "engine: device={} models={} ({}) texconv={}".format(
            engine.device,
            a.models,
            engine.describe() or "NO MODELS",
            texio.TEXCONV if texio.use_texconv() else "numpy",
        )
    )
    check_sysmem_fallback(engine)
    if not check_encoder(a):
        return 2
    # Order is what the caller asked for; duplicates are dropped so a repeated --family does not
    # re-walk a family that is already finished (the second pass costs a query per role, not work,
    # but it makes the log look like something went wrong).
    families = (
        list(dict.fromkeys(a.family))
        if a.family
        else [f for f, (t, _) in roles.POLICY.items() if t]
    )
    if a.profile == "everyday" and not a.family:
        families = [f for f in families if f not in ("monster", "demihuman")]
    # Before the first texture, not after the last: see check_disk.
    if not check_disk(a, man, families):
        return 2
    print(
        "Output estimate above is a legacy upper bound; everyday writes only the selected product per resource."
    )
    gamever = kb.sqpack.game_version(kb.game().sqpack)
    print(f"game version: {gamever} (recorded on every row this run finishes)")
    since = None
    if a.since:
        since = time.mktime(time.strptime(a.since, "%Y-%m-%d"))
        print(
            f"only the patch delta: rows enumerated on/after {a.since}, plus rows a fingerprint check re-queued"
        )
    t0 = time.time()
    counts = {"done": 0, "failed": 0}
    uld_stats = {"sheets": 0, "parts": 0}
    interrupted = False
    # Built once, from every ui/uld sheet the manifest knows, because a sheet's rectangles are
    # frequently defined by ANOTHER widget's .uld -- Parameter_Gauge.tex has none of its own.
    uld_index = {}
    if "ui-uld" in families and not a.no_uld_parts:
        stems = [
            r[0].rsplit("/", 1)[-1][:-4]
            for r in man.db.execute("SELECT path FROM tex WHERE family='ui-uld'")
        ]
        uld_index = uldparts.build_index(
            kb.game(),
            stems,
            pathlist=paths.newest_pathlist(),
            log=lambda m: print("  " + m, flush=True),
        )
    # The encode (texconv subprocess + file writes) overlaps the model pass of the textures that
    # follow. Textures come off the model into `staging`; a full batch goes to a worker as ONE
    # encode job; results are reaped in order, so the manifest only ever marks a texture done after
    # every tier of it is on disk. The batch is the point: a texconv call is ~0.3 s of fixed cost
    # (process, D3D11 device, shader compile) and this queue is mostly textures whose actual encode
    # is a small fraction of that -- see texio._texconv_many and scripts/bench_texconv.py.
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, a.encode_workers)
    pool = ThreadPoolExecutor(max_workers=workers)
    cap_n, cap_bytes = max(1, a.encode_batch), max(1, a.encode_mb) * 1024 * 1024
    staging = []  # [(path, job, info, t1, t2, srch)] -- through the model, not yet submitted
    inflight = []  # [(future, [(path, info, t1, t2, srch), ...])] -- submitted, in order

    def encode_batch(jobs):
        """Encode one batch of textures and write their tiers.

        jobs: [(path, mod, top, img, fmt_out, attr, keep_mips)] -> one entry per job, in order:
        (tiers written, encode seconds per texture, batch size), or the exception for that job.
        """
        # Timed HERE, in the worker, not at reap. Reap is deliberately behind so the encode of one
        # batch overlaps the model pass of the next, and a figure read at reap time would describe
        # the model, not the encode. The number has to describe the thing it names.
        t_enc = time.time()
        items, plans = [], []
        for path, mod, top, img, fmt_out, attr, keep_mips in jobs:
            offsets = {
                tier: k
                for k, tier in enumerate(roles.tiers_below(top))
                if a.profile == "legacy" or tier == top
            }
            want = [k for tier, k in offsets.items() if not a.tiers or tier in a.tiers]
            items.append((img, fmt_out, attr, want, keep_mips))
            plans.append((path, mod, offsets))
        texs = texio.encode_tiers_many(items)
        written = []
        for (path, mod, offsets), tex in zip(plans, texs, strict=True):
            if isinstance(tex, Exception):
                written.append(tex)
                continue
            try:
                wrote = []
                for tier, k in offsets.items():
                    if k not in tex:
                        continue
                    dst = os.path.join(a.out, mod, packer.file_rel(tier, path))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    pathlib.Path(dst).write_bytes(tex[k])
                    wrote.append(tier)
                written.append(wrote)
            except Exception as e:
                written.append(e)
        per = (time.time() - t_enc) / len(jobs)
        return [w if isinstance(w, Exception) else (w, per, len(jobs)) for w in written]

    def flush():
        """Hand everything in staging to a worker as one batch."""
        if not staging:
            return
        jobs = [job for _, job, *_ in staging]
        meta = [(path, info, t1, t2, srch) for path, _, info, t1, t2, srch in staging]
        inflight.append((pool.submit(encode_batch, jobs), meta))
        staging.clear()

    def submit(path, job, info, t1, t2, srch):
        """A texture is through the model: stage it, flush when the batch is full, reap."""
        staging.append((path, job, info, t1, t2, srch))
        # Capped by bytes as well as by count. The image is the UPSCALED one: a 4x pass over a
        # 2048 source is 256 MiB, and it is written to scratch uncompressed for texconv, so the
        # cap bounds both RAM and scratch. Icons batch by the dozen; the big ones go alone, where
        # the fixed cost was never the problem.
        if len(staging) >= cap_n or sum(j[3].nbytes for _, j, *_ in staging) >= cap_bytes:
            flush()
        reap()

    def finish(path, info, t1, t2, srch, res):
        """Record one texture's outcome in the manifest."""
        if not isinstance(res, Exception):
            wrote, enc_s, nb = res
            note = "models:" + engine.describe()
            if engine.missing:
                note += " fallback:" + ",".join(sorted(engine.missing))
            # The source hash is written in the same statement that marks the row done, so the
            # stamp always describes the bytes these files were actually made from. Stamping
            # separately afterwards would leave a window where a row claims done with no hash
            # (or worse, a hash taken after a patch landed mid-run).
            man.set_status(
                path,
                "done",
                tiers=",".join(wrote),
                note=note[:400],
                srchash=srch or "",
                srcver=gamever if srch else "",
            )
            man.db.execute(
                "UPDATE tex SET recipe=? WHERE path=?",
                (
                    "profile="
                    + a.profile
                    + ";policy="
                    + release.POLICY_VERSION
                    + ";models="
                    + engine.describe(),
                    path,
                ),
            )
            counts["done"] += 1
            if a.verbose or counts["done"] <= 5:
                batch = f" (batch of {nb})" if nb > 1 else ""
                print(f"  {info}  model {t2 - t1:.1f}s  encode {enc_s:.1f}s{batch}")
        elif interrupted:
            # NOT A FAILURE. On Windows a Ctrl+C goes to the whole process GROUP, so the texconv
            # child that happened to be encoding when the key was pressed is killed too and
            # returns non-zero. Recording that as 'failed' contradicts this function's own
            # contract -- an interrupted texture is supposed to stay 'planned' and simply be
            # redone -- and it is worse than cosmetic: 'failed' rows are not 'planned' rows, so
            # the texture silently never comes back unless somebody thinks to run
            # `requeue --failed`. Exactly one row per run, one per Ctrl+C, which is precisely how
            # three of them accumulated.
            man.set_status(path, "planned", note="interrupted mid-encode; will be redone")
            print(
                "  (interrupted during {} -- left planned, not failed)".format(
                    path.rsplit("/", 1)[-1]
                )
            )
        else:
            counts["failed"] += 1
            man.set_status(path, "failed", note=str(res)[:400])
            if a.verbose:
                traceback.print_exception(res)
        man.commit()
        n = counts["done"] + counts["failed"]
        if n % 25 == 0:
            el = time.time() - t0
            print(
                "  %d done, %d failed, %.0fs (%.1f s/texture)"
                % (counts["done"], counts["failed"], el, el / n)
            )

    def reap(all_of_them=False):
        # As many batches stay in flight as there are workers; the oldest beyond that is waited
        # for. With one worker that is: the batch being encoded, while the next one fills.
        while inflight and (all_of_them or len(inflight) > workers):
            fut, meta = inflight.pop(0)
            try:
                results = fut.result()
            except Exception as e:
                results = [e] * len(meta)
            for (path, info, t1, t2, srch), res in zip(meta, results, strict=True):
                finish(path, info, t1, t2, srch, res)

    def drain():
        """Everything through the model is encoded and recorded before this returns."""
        flush()
        reap(True)

    # Ctrl+C is a normal way to stop a run that will take days, so it exits the way --budget does:
    # whatever is already through the model is finished and committed, and nothing half-written is
    # ever marked done (a texture is committed only after every tier of it is on disk, so an
    # interrupted one stays 'planned' and is simply redone).
    try:

        def run_group(group, role):
            """One model call per stage for a run of same-sized UI textures (see roles.do_ui_batch).

            Decode failures drop out of the group and are recorded individually, so one bad file
            cannot take the batch with it.
            """
            ready = []
            for path, mod, top, w, h, fmt in group:
                try:
                    hdr, rgba = texio.read(texio.read_raw(path))
                    ready.append((path, mod, top, w, h, fmt, hdr, rgba))
                except Exception as e:
                    counts["failed"] += 1
                    man.set_status(path, "failed", note=str(e)[:400])
            if not ready:
                man.commit()
                return
            w0, h0, top0 = ready[0][3], ready[0][4], ready[0][2]
            print(
                "  ... %d x %dx%d -> %s through the model" % (len(ready), w0, h0, top0),
                flush=True,
            )
            t1 = time.time()
            imgs = roles.process_top_batch(engine, role, fam, [r[7] for r in ready], ready[0][2])
            t2 = time.time()
            per = (t2 - t1) / len(ready)
            for (path, mod, top, w, h, fmt, hdr, _), img in zip(ready, imgs, strict=True):
                fmt_out = texio.out_format(hdr.format_name, role)
                submit(
                    path,
                    (path, mod, top, img, fmt_out, hdr.attributes, hdr.mip_count > 1),
                    "%s %dx%d %s -> %s (model batch of %d)"
                    % (path.rsplit("/", 1)[-1], w, h, fmt, top, len(ready)),
                    t2 - per,
                    t2,
                    src_fingerprint(path),
                )

        for family in families:
            print(f"\n>>> Starting family: {family}", flush=True)
            for role in ("normal", "mask", "color", "icon", "ui"):
                rows = man.rows(
                    family=family,
                    role=role,
                    status="planned",
                    limit=a.limit,
                    since=since,
                )
                # UI and icons: gather runs of the same size and hand them to the model together.
                # ui-uld opts out when per-part upscaling is on, because the per-part path lives on
                # the single-row branch below. Nothing is lost: the batch is capped by input pixels
                # and a uld sheet is large enough that its group size was already 1.
                batchable = role in ("icon", "ui") and a.icon_batch > 1
                if family == "ui-uld" and not a.no_uld_parts:
                    batchable = False
                if batchable:
                    # Bucket by size across the whole family rather than batching consecutive rows:
                    # icons alternate 40x40 / 80x80 (`000001.tex`, `000001_hr1.tex`), so a
                    # run-length grouping produced batches of one to three and saved nothing.
                    buckets = {}
                    for path, fam, _part, role_, w, h, fmt, _mips, _status, _tiers in rows:
                        mod = packer.mod_for_family(fam)
                        top = release.generation_tier(fam, role_, path, w, h, a.top, a.profile)
                        if mod is None or top is None:
                            man.set_status(path, "skipped", note="policy")
                            continue
                        buckets.setdefault((w, h, top), []).append((path, mod, top, w, h, fmt))
                    if buckets:
                        px = sum(k[0] * k[1] * len(v) for k, v in buckets.items())
                        big = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:3]
                        print(
                            "%s/%s: %d planned in %d size groups, %.0f Mpx in — largest groups %s"
                            % (
                                family,
                                role,
                                sum(len(v) for v in buckets.values()),
                                len(buckets),
                                px / 1e6,
                                ", ".join("%dx%d x%d" % (k[0], k[1], len(v)) for k, v in big),
                            ),
                            flush=True,
                        )
                    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
                        items = buckets[key]
                        # The group is decoded before the model sees it, so cap it by input pixels
                        # as well as by count: 32 icons is 50 k pixels, 32 of a 2560x720 uld sheet
                        # would be 59 M. Big sheets simply fall back to one at a time, where they
                        # were anyway.
                        w_, h_ = key[0], key[1]
                        n = max(1, min(a.icon_batch, (2 * 1024 * 1024) // max(1, w_ * h_)))
                        for s in range(0, len(items), n):
                            if a.budget and time.time() - t0 > a.budget:
                                break
                            run_group(items[s : s + n], role)
                        if a.budget and time.time() - t0 > a.budget:
                            break
                    drain()
                    man.commit()
                    if a.budget and time.time() - t0 > a.budget:
                        print(
                            "budget reached: %d done, %d failed — rerun to resume"
                            % (counts["done"], counts["failed"])
                        )
                        return 3
                    continue
                for path, fam, _part, role_, w, h, fmt, _mips, _status, _tiers in rows:
                    if a.budget and time.time() - t0 > a.budget:
                        drain()
                        print(
                            "budget reached: %d done, %d failed — rerun to resume"
                            % (counts["done"], counts["failed"])
                        )
                        return 3
                    mod = packer.mod_for_family(fam)
                    top = release.generation_tier(fam, role_, path, w, h, a.top, a.profile)
                    if mod is None or top is None:
                        man.set_status(path, "skipped", note="policy")
                        continue
                    try:
                        t1 = time.time()
                        raw = texio.read_raw(path)
                        hdr, rgba = texio.read(raw)
                        img = roles.process_top(engine, role_, fam, rgba, hdr.format_name, top)
                        # UI atlases: redo each ULD sprite from its own padded crop, so the model's
                        # receptive field cannot reach across a seam between two abutting sprites.
                        # See uldparts -- JobHudXBM1 alone has 93 touching pairs among 49 parts.
                        if fam == "ui-uld" and not a.no_uld_parts and roles.TIER_SCALE[top] > 1:
                            rects = uldparts.rects_for(
                                kb.game(),
                                path,
                                hdr.width,
                                hdr.height,
                                index=uld_index or None,
                            )
                            if rects:
                                uld_stats["sheets"] += 1
                                uld_stats["parts"] += len(rects)
                                img = uldparts.upscale_by_parts(
                                    lambda cs, r=role_, f=fam, t=top: roles.process_top_batch(
                                        engine, r, f, cs, t
                                    ),
                                    rgba,
                                    roles.TIER_SCALE[top],
                                    rects,
                                    img,
                                )
                        t2 = time.time()
                        fmt_out = texio.out_format(hdr.format_name, role_)
                        submit(
                            path,
                            (path, mod, top, img, fmt_out, hdr.attributes, hdr.mip_count > 1),
                            "%s %dx%d %s -> %s" % (path.rsplit("/", 1)[-1], w, h, fmt, top),
                            t1,
                            t2,
                            src_fingerprint(path),
                        )
                    except Exception as e:
                        counts["failed"] += 1
                        man.set_status(path, "failed", note=str(e)[:400])
                        man.commit()
                        if a.verbose:
                            traceback.print_exc()
            drain()
            man.commit()
        drain()
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted — finishing the textures already through the model...")
        try:
            drain()
        except KeyboardInterrupt:
            print("  (second Ctrl+C: dropping the queue; those textures stay planned)")
        man.commit()
        pool.shutdown(wait=False)
        el = time.time() - t0
        n = counts["done"] + counts["failed"]
        rate = " (%.1f s/texture)" % (el / n) if n else ""
        if not n:
            rate = " — nothing finished; the first group was still in the model"
        print(
            "stopped: %d done, %d failed in %.0fs%s — rerun the same command to resume"
            % (counts["done"], counts["failed"], el, rate)
        )
        return 3
    pool.shutdown()
    man.commit()
    print(
        "finished: %d done, %d failed in %.0fs; model slots without a file: %s"
        % (
            counts["done"],
            counts["failed"],
            time.time() - t0,
            sorted(engine.missing) or "none",
        )
    )
    if uld_stats["sheets"]:
        print(
            "  ui/uld: %d sheet(s) composited from %d ULD part rectangles instead of upscaled whole"
            % (uld_stats["sheets"], uld_stats["parts"])
        )
    return 0


def cmd_pack(a):
    man = mf.Manifest(a.db)
    written = packer.pack(man, a.out)
    for mod, tiers in written.items():
        print("%-45s %s" % (mod, tiers))
    if a.penumbra_config:
        p = packer.merge_penumbra(a.penumbra_config, list(written))
        print(
            "merged into",
            p,
            "and the Default collection — restart the game (or Rediscover Mods)",
        )


def cmd_qa(a):
    """Contamination + normal-length + drift checks on a sample of finished textures."""
    man = mf.Manifest(a.db)
    rows = man.rows(status="done")
    if a.family:
        rows = [r for r in rows if r[1] in a.family]
    import random

    random.seed(1)
    sample = random.sample(rows, min(a.samples, len(rows)))
    for path, family, _part, role, w, h, fmt, _mips, _status, tiers in sample:
        top = (tiers or "").split(",")[0]
        mod = packer.mod_for_family(family)
        f = os.path.join(a.out, mod, packer.file_rel(top, path))
        if not os.path.isfile(f):
            continue
        hdr, out = texio.read(pathlib.Path(f).read_bytes())
        _, src = texio.read(path)
        s = out.shape[0] // src.shape[0]
        msg = [
            "%s %s %s->%s %dx%d→%dx%d"
            % (family, role, fmt, hdr.format_name, w, h, out.shape[1], out.shape[0])
        ]
        if role == "normal":
            nx, ny = out[..., 0] / 127.5 - 1, out[..., 1] / 127.5 - 1
            over = float((nx * nx + ny * ny > 1.02).mean())
            msg.append("XY>1: %.3f%%" % (100 * over))
        # per-channel low-frequency drift vs the source (contamination shows as drift in a channel that was flat)
        from .processing.engine import box_down

        srcf = src.astype(np.float32) / 255
        outf = box_down(out.astype(np.float32) / 255, s) if s > 1 else out.astype(np.float32) / 255
        outf = outf[: srcf.shape[0], : srcf.shape[1]]
        d = np.abs(outf - srcf).mean((0, 1))
        msg.append("mean|Δ| R {:.3f} G {:.3f} B {:.3f} A {:.3f}".format(*tuple(d)))
        print("  ".join(msg), path)


def cmd_modup(a):
    from . import modtex

    engine = Engine(a.models, device=a.device, allow_fallback=not a.strict)
    print(
        "engine: device={} models={} texconv={}".format(
            engine.device, a.models, texio.TEXCONV if texio.use_texconv() else "numpy"
        )
    )
    if not check_encoder(a):
        return 2
    twins = {}
    for mod in a.mod:
        dst, _ = modtex.upscale_mod(
            mod, a.out, engine, icon_scale=a.icon_scale, sheet_scale=a.sheet_scale
        )
        twins[os.path.basename(os.path.normpath(mod))] = dst
    if a.penumbra_config:
        print(
            "merged into",
            packer.merge_icon_twins(a.penumbra_config, twins),
            "— restart the game or Rediscover Mods",
        )
    if engine.missing:
        print(
            f"NOTE: Lanczos fallback was used for slots {sorted(engine.missing)} — put the models in {a.models} for the real thing"
        )


# ---------------------------------------------------------------- requeue / audit
# Formats tools/texdecode.py can decode. Anything else reaching a processed role is a texture the
# run would fail on, which is what `audit` is for.
# Where `run` writes when nothing says otherwise. Overridable by CLARITY_OUT or --out.
DEFAULT_OUT = os.path.join(paths.PROJECT, "build-output", "out")
OUT_DEFAULT = os.environ.get("CLARITY_OUT", DEFAULT_OUT)
# Penumbra's own config dir on this machine; `pack` merges the mods into the Default collection there.
PENUMBRA_CONFIG_DEFAULT = os.environ.get(
    "CLARITY_PENUMBRA_CONFIG",
    os.path.join(os.environ.get("APPDATA", ""), "XIVLauncher", "pluginConfigs", "Penumbra"),
)

DECODABLE = {
    "BC1",
    "BC2",
    "BC3",
    "BC4",
    "BC5",
    "BC7",
    "BC6H",
    "B8G8R8A8",
    "B8G8R8X8",
    "B4G4R4A4",
    "B5G5R5A1",
    "L8",
    "A8",
}


def cmd_reclassify(a):
    """Re-run classify() over rows already in the manifest and write back what changed.

    `plan` inserts with INSERT OR IGNORE, so a row's family/part/role is whatever classify() said
    the first time it was seen and nothing revisits it. Every later change to the classifier --
    a new suffix rule, a family split, the loading screens moving from `other` to `ui` -- applies
    only to textures scanned after it, which is a manifest that quietly disagrees with the code.

    Rows the header marked `skip` are left alone: that verdict came from the texture (arrays,
    cubes, float formats, depth) rather than from the path, and classify() cannot re-derive it
    without re-reading the file. Nothing here changes status, so a `done` row stays done -- pair
    with `requeue` if the new role should send it through the model again.
    """
    man = mf.Manifest(a.db)
    sql, args = "1=1", []
    if a.family:
        sql += " AND family IN ({})".format(",".join("?" * len(a.family)))
        args += a.family
    if a.path_like:
        sql += " AND path LIKE ?"
        args.append(a.path_like)

    changes = []
    for path, family, part, role, w, h, fmt, ttype in man.db.execute(
        "SELECT path, family, part, role, w, h, fmt, ttype FROM tex WHERE " + sql, args
    ):
        if role == "skip":
            continue
        # CLASSIFY WITH THE HEADER FACTS, NOT WITHOUT THEM. Calling classify(path, None) re-ran only
        # the half of the classifier that reads the path, so every rule that needs the texture was
        # silently reversed: a big ui/icon map is promoted from role `icon` to `ui` by
        # `hdr.width > 512`, and with no header that promotion does not happen, so reclassify saw a
        # "change" from ui back to icon and wrote it. The manifest already stores w, h, fmt and
        # ttype -- everything classify() consults -- so hand it those instead of nothing.
        nf, np_, nr = mf.classify(path, mf.StoredHeader(w, h, fmt, ttype))
        if (nf, np_, nr) != (family, part, role):
            changes.append((path, family, part, role, nf, np_, nr))

    if not changes:
        print("nothing to reclassify")
        return 0

    # The label carries `part` as well as family/role, because the housing split moves a great many
    # rows whose family and role do not change at all -- bg/ rows gaining part 'fld', 'dun', 'twn'.
    # Without it the summary reads "bg/color -> bg/color 61,344", which looks like a no-op loop.
    seen = {}
    for _p, f, pt, r, nf, npt, nr in changes:
        k = "{}[{}]/{} -> {}[{}]/{}".format(f, pt or "-", r, nf, npt or "-", nr)
        seen[k] = seen.get(k, 0) + 1
    for k, n in sorted(seen.items(), key=lambda x: -x[1])[:30]:
        print("  %-58s %7d" % (k, n))
    if len(seen) > 30:
        print("  ... and %d more kinds of change" % (len(seen) - 30))
    print("  %d row(s)" % len(changes))

    if a.dry_run:
        print("  dry run, nothing written")
        return 0

    man.db.executemany(
        "UPDATE tex SET family=?, part=?, role=? WHERE path=?",
        [(nf, npt, nr, p) for p, _f, _pt, _r, nf, npt, nr in changes],
    )
    man.db.commit()
    print("  written")
    if a.sync_status:
        n, why = mf.sync_skipped(man)
        print("  status sync: %d planned row(s) marked skipped" % n)
        for k, c_ in why.most_common():
            print("    %-52s %6d" % (k[:52], c_))
    return 0


def cmd_requeue(a):
    """Put rows back to 'planned' so the next run redoes them."""
    man = mf.Manifest(a.db)
    where, args = [], []
    if a.failed:
        where.append("status = 'failed'")
    if a.skipped:
        where.append("status = 'skipped'")
    if a.old_recipe:
        # rows finished before the model registry was in place carry no models: note
        where.append("(status = 'done' AND (note IS NULL OR note NOT LIKE 'models:%'))")
    if a.without_model:
        where.append("(status = 'done' AND (note IS NULL OR note NOT LIKE ?))")
        args.append("%" + a.without_model + "%")
    if a.done:
        # THE SELECTOR FOR A PROCESSING CHANGE. --old-recipe and --without-model both ask the
        # models: note a question, so they can only see changes of MODEL. A change to how a role is
        # processed -- the uldparts padding fix, a channel handled differently, a resampling change
        # -- leaves the note identical and is invisible to both, and there was no way to say "redo
        # these finished rows" without hand-writing UPDATEs against the database, which is exactly
        # what the README no longer recommends.
        #
        # Guarded rather than free: with no filter this is "throw away every finished texture and
        # start the 100-GPU-hour job again", which nobody means to type.
        if not (a.family or a.role or a.path_like):
            print("--done needs at least one of --family / --role / --path-like:")
            print("  on its own it would requeue every finished texture in the manifest.")
            return 2
        where.append("status = 'done'")
    if not where:
        print("nothing selected: pass --failed, --skipped, --old-recipe, --without-model NAME,")
        print("or --done with --family/--role/--path-like (for a processing change)")
        return 2
    sql = "(" + " OR ".join(where) + ")"
    if a.family:
        sql += " AND family IN ({})".format(",".join("?" * len(a.family)))
        args += a.family
    if a.role:
        sql += " AND role IN ({})".format(",".join("?" * len(a.role)))
        args += a.role
    if a.path_like:
        sql += " AND path LIKE ?"
        args.append(a.path_like)
    rows = man.db.execute(
        "SELECT family, role, status, COUNT(*) FROM tex WHERE "
        + sql
        + " GROUP BY family, role, status ORDER BY 4 DESC",
        args,
    ).fetchall()
    total = sum(r[3] for r in rows)
    for fam, role, status, n in rows[:12]:
        print("  %-12s %-7s %-8s %7d" % (fam, role, status, n))
    if len(rows) > 12:
        print("  ... %d more group(s)" % (len(rows) - 12))
    print("total: %d row(s)" % total)
    if a.dry_run or not total:
        print("(dry run - nothing written)" if a.dry_run else "(nothing to do)")
        return 0
    # Keep the `changed at <version>` marker: `run --since` selects the patch delta by
    # it, so a changed row that failed and is re-queued here must stay in that delta.
    # Every other note describes the done/failed/skipped state being undone.
    man.db.execute(
        "UPDATE tex SET status='planned', tiers='', "
        "note=CASE WHEN note LIKE 'changed at %' THEN note ELSE '' END WHERE " + sql,
        args,
    )
    man.commit()
    print("re-queued %d row(s); rerun `clarity run` to redo them" % total)
    return 0


# ---------------------------------------------------------------- fingerprint
def src_fingerprint(path, gd=None):
    """Content hash of the game's copy of `path`, or None if it is not in the index."""
    gd = gd or kb.game()
    loc = gd.locate(path)
    if loc is None:
        return None
    try:
        return kb.sqpack.entry_fingerprint(*loc)
    except Exception:
        return None


def _bar(done, total, width=44):
    if total <= 0:
        return "[%s]" % ("?" * width)
    f = done / total
    n = int(f * width)
    return "[{}{}] {:5.1f}%".format("#" * n, "." * (width - n), 100 * f)


def cmd_fingerprint(a):
    """Record, or check, what the game's copy of every source texture looked like.

    The problem this exists for: a finished upscale is a *derivative* of a specific version of a
    specific .tex, and nothing in the manifest used to remember which one. When a patch retouches a
    texture, the row still says `done`, the file is still on disk, and the mod quietly keeps
    serving an upscale of the previous artwork -- forever, because `run` only ever looks at rows
    marked `planned`. There is no error and nothing to notice.

    So: `--stamp` (the default) writes today's hash onto rows that have one missing, and `--check`
    recomputes and reports the difference. Run stamp before a patch and check after it, and the
    delta is exactly the list of textures to redo. `--check --requeue` turns that list back into
    work.

    This is deliberately a standing capability rather than a one-off for a particular patch. 7.56 is
    not the last one -- there is a Gold Saucer patch coming, the typing minigame, regional party
    finder, and hotfixes have a long history of carrying unannounced asset changes along with
    whatever they were actually for. Every one of those is another chance to end up with upscales of
    art that no longer exists, so the check wants to be a thing you can run on any Tuesday, cheaply,
    and not a thing that has to be rebuilt each time.
    """
    man = mf.Manifest(a.db)
    gd = kb.game()
    version = kb.sqpack.game_version(gd.sqpack)

    if a.export:
        return export_fingerprints(man, a.export if a.export != "-" else paths.FINGERPRINTS)

    if a.log:
        rows = man.snapshots()
        if not rows:
            print("no fingerprint runs recorded yet")
            return 0
        print(
            "%-19s %-22s %-7s %8s %8s %7s  %s"
            % ("when", "game version", "mode", "hashed", "changed", "gone", "note")
        )
        for ts, ver, mode, nh, nc, ng, _nn, note in rows:
            print(
                "%-19s %-22s %-7s %8d %8d %7d  %s"
                % (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)),
                    ver,
                    mode,
                    nh,
                    nc or 0,
                    ng or 0,
                    note,
                )
            )
        return 0

    where, args = [], []
    if not a.all:
        where.append("status = 'done'")
    if a.family:
        where.append("family IN ({})".format(",".join("?" * len(a.family))))
        args += a.family
    if a.path_like:
        where.append("path LIKE ?")
        args.append(a.path_like)
    if a.check:
        where.append("srchash != ''")  # nothing to compare against otherwise
    elif not a.restamp:
        where.append("srchash = ''")  # stamp only what has never been stamped
    sql = " AND ".join(where) or "1=1"

    rows = man.db.execute(
        "SELECT path, srchash, srcver, family, status FROM tex WHERE " + sql + " ORDER BY path",
        args,
    ).fetchall()
    total = len(rows)
    mode = "check" if a.check else ("restamp" if a.restamp else "stamp")
    print(f"game version: {version}")
    print("%s %d row(s)%s" % (mode, total, "" if a.all else " (status=done; --all for every row)"))
    if not total:
        if mode == "stamp":
            print("  every selected row already carries a hash — use --restamp to take them again,")
            print("  or --check to compare them against the install as it stands now")
        else:
            print(
                "  nothing stamped yet — run `clarity fingerprint` first, there is no 'before' to compare to"
            )
        return 0

    t0 = time.time()
    changed, gone, _unstamped, ok = [], [], 0, 0
    writes = []
    for i, (path, old, oldver, family, _status) in enumerate(rows):
        h = src_fingerprint(path, gd)
        if h is None:
            gone.append((path, family))
        elif a.check:
            if h != old:
                changed.append((path, family, old, h, oldver))
            else:
                ok += 1
        else:
            writes.append((h, version, path))
        if (i + 1) % 500 == 0 or i + 1 == total:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            eta = (total - i - 1) / max(rate, 1e-6)
            print(
                "\r  %s %7d/%-7d  %5.0f/s  eta %s   "
                % (_bar(i + 1, total), i + 1, total, rate, _fmt_eta(eta)),
                end="",
                flush=True,
            )
    print()

    if writes:
        man.db.executemany("UPDATE tex SET srchash=?, srcver=? WHERE path=?", writes)
        man.commit()
        print("  stamped %d row(s) at %s" % (len(writes), version))
    if gone:
        print("\n  %d path(s) no longer in the index (removed or renamed by a patch):" % len(gone))
        for p, fam in gone[:15]:
            print("    %-10s %s" % (fam, p))
        if len(gone) > 15:
            print("    ... %d more" % (len(gone) - 15))

    if a.check:
        print("\n  unchanged %d, CHANGED %d, gone %d" % (ok, len(changed), len(gone)))
        if changed:
            byfam = {}
            for _p, fam, _o, _n, _ov in changed:
                byfam[fam] = byfam.get(fam, 0) + 1
            print("\n  changed by family:")
            for fam, n in sorted(byfam.items(), key=lambda x: -x[1]):
                print("    %-14s %6d" % (fam, n))
            print("\n  first %d:" % min(15, len(changed)))
            for p, fam, o, n, ov in changed[:15]:
                print("    %-10s %s  %s(%s) -> %s" % (fam, p, o[:12], ov or "?", n[:12]))
            if a.requeue:
                man.db.executemany(
                    "UPDATE tex SET status='planned', tiers='', note=? WHERE path=?",
                    [("changed at " + version, p) for p, _f, _o, _n, _ov in changed],
                )
                man.commit()
                print(
                    "\n  re-queued %d row(s) — `clarity run` will redo them, and will re-stamp as it goes"
                    % len(changed)
                )
            else:
                print("\n  nothing written (add --requeue to send these back through the model)")
        print(
            "\n  note: this compares rows the manifest already knows about. Content a patch *adds*"
        )
        print("        shows up only when the manifest is re-enumerated — rerun `clarity plan`")
        print(
            "        (--chara --icons, and --pathlist with a fresh ResLogger list for bg/ and ui/uld/)."
        )

    man.snapshot(
        version,
        mode,
        len(writes) if not a.check else ok + len(changed),
        len(changed),
        len(gone),
        0,
        note=("requeued" if (a.check and a.requeue and changed) else ""),
    )
    print("\n  %.0fs" % (time.time() - t0))
    return 0


def export_fingerprints(man, out):
    """Write every stamped row as `path<TAB>srchash<TAB>srcver<TAB>status`, sorted, to hash-manifests.

    The database is a 100 MB+ binary that git cannot diff; this text file is the same fact in a
    form it can, and it is what `git_archive.py verify`-style checks and a future rebuild of the
    database can start from. Deterministic order so a re-export of unchanged state is a no-op diff.
    """
    rows = man.db.execute(
        "SELECT path, srchash, srcver, status FROM tex WHERE srchash != '' ORDER BY path"
    ).fetchall()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "# clarity texture fingerprints: path, BLAKE2b of the compressed sqpack entry, game version it was taken on, status\n"
        )
        for path, h, ver, status in rows:
            f.write(f"{path}\t{h}\t{ver}\t{status}\n")
    os.replace(tmp, out)
    print("exported %d fingerprint(s) -> %s" % (len(rows), out))
    return 0


def _fmt_eta(s):
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return "%.0fm" % (s / 60)
    return "%.1fh" % (s / 3600)


def cmd_audit(a):
    """Report the classes of texture that the pipeline treats specially, or cannot treat at all."""
    man = mf.Manifest(a.db)

    def q(s, *p):
        return man.db.execute(s, p).fetchall()

    print("status:", dict(q("SELECT status, COUNT(*) FROM tex GROUP BY status")))
    roles_in = ",".join(f"'{r}'" for r in mf.PROCESSED_ROLES)

    print("\nformats with no decoder, in a role the run processes:")
    bad = q(
        "SELECT fmt, family, role, COUNT(*) FROM tex WHERE role IN ({}) AND fmt NOT IN ({})"
        " GROUP BY fmt, family, role ORDER BY 4 DESC".format(
            roles_in, ",".join(f"'{f}'" for f in sorted(DECODABLE))
        )
    )
    for fmt, fam, role, n in bad:
        print("  %-14s %-10s %-6s %7d   <-- would fail" % (fmt, fam, role, n))
    if not bad:
        print("  none")

    print("\nsmaller than the 8 px pad (replicate padding, no tiling):")
    for w, h, fmt, role, n in q(
        "SELECT w, h, fmt, role, COUNT(*) FROM tex WHERE (w < 8 OR h < 8)"
        f" AND role IN ({roles_in}) GROUP BY w, h, fmt, role ORDER BY 5 DESC LIMIT 6"
    ):
        print("  %2dx%-3d %-6s %-6s %6d" % (w, h, fmt, role, n))

    # The tier chain stays exact on these: every tier is the top divided by a power of two, and the
    # top is the source times 4 or 2, so the halvings land on whole pixels. What they do exercise is
    # the encoder's partial last block.
    print("\nnot a multiple of 4 (the block size; the encoder pads the last block):")
    for w, h, role, n in q(
        "SELECT w, h, role, COUNT(*) FROM tex WHERE (w % 4 OR h % 4)"
        f" AND role IN ({roles_in}) GROUP BY w, h, role ORDER BY 4 DESC LIMIT 6"
    ):
        print("  %dx%-4d %-6s %6d" % (w, h, role, n))

    print("\nnon-2D (skipped by classification, never reaches the run):")
    for t, n in q(
        "SELECT ttype, COUNT(*) FROM tex WHERE ttype != '2D' GROUP BY ttype ORDER BY 2 DESC"
    ):
        print("  %-8s %7d" % (t, n))

    print("\nno mip chain in the source (we emit one; the game reads the header):")
    print(
        "  mips=1  %d of %d"
        % (
            q("SELECT COUNT(*) FROM tex WHERE mips <= 1")[0][0],
            q("SELECT COUNT(*) FROM tex")[0][0],
        )
    )

    print("\nat or above the %d output cap (top tier drops a step):" % roles.MAX_EDGE_OUT)
    for w, h, n in q(
        "SELECT w, h, COUNT(*) FROM tex WHERE MAX(w, h) * 4 > %d AND role IN (%s)"
        " GROUP BY w, h ORDER BY 3 DESC LIMIT 6" % (roles.MAX_EDGE_OUT, roles_in)
    ):
        print("  %dx%-5d %6d" % (w, h, n))

    print("\ndone rows by recipe:")
    for note, n in q(
        "SELECT CASE WHEN note LIKE 'models:%' THEN 'registry recipe' ELSE 'pre-registry' END,"
        " COUNT(*) FROM tex WHERE status='done' GROUP BY 1"
    ):
        print("  %-18s %7d" % (note, n))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="clarity")
    ap.add_argument("--db", default=paths.DB, help="manifest database (default: %(default)s)")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug-level diagnostics on stderr (per-texture timings, skipped inputs, tracebacks)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument(
        "--pathlist",
        default=paths.newest_pathlist(),
        help="ResLogger CurrentPathList (default: newest in analysis-specimens\\reslogger)",
    )
    p.add_argument("--chara", action="store_true")
    p.add_argument("--icons", action="store_true")
    p.add_argument("--top")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("estimate")
    p.add_argument("--top")
    p.set_defaults(fn=cmd_estimate)
    release.add_parser(sub)
    p = sub.add_parser("run")
    p.add_argument(
        "--profile",
        choices=["everyday", "legacy"],
        default="everyday",
        help="everyday writes one selected product; legacy writes all tiers",
    )
    # The output has been the same Penumbra root on every run this tool has ever done, so it is a
    # default rather than a required argument. CLARITY_OUT overrides it without editing anything,
    # and --out still wins over both -- this only removes the typing, not the choice.
    p.add_argument(
        "--out",
        default=OUT_DEFAULT,
        help="Penumbra mod root to write into (default: %(default)s)",
    )
    p.add_argument("--family", action="append")
    p.add_argument("--top", help="force the top tier: 4x | 2x | native")
    p.add_argument("--tiers", nargs="*")
    p.add_argument("--budget", type=float, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--strict-disk",
        action="store_true",
        help="stop rather than warn when the queued output will not fit on the destination volume",
    )
    p.add_argument(
        "--no-uld-parts",
        action="store_true",
        help="upscale ui/uld sheets whole instead of per ULD part (the old behaviour: "
        "faster, but the model reaches across the seams between abutting sprites)",
    )
    p.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="only the patch delta: rows added by a `plan` on/after this date, plus rows a fingerprint check re-queued",
    )
    p.add_argument("--models", default=paths.MODELS)
    p.add_argument("--device")
    # 512, not 256. Every run on this machine passed --tile 512 by hand, so the default was
    # only ever the thing you had to remember to override. A 4070 Super holds a 512 tile
    # comfortably and it means fewer seams and fewer model calls per texture.
    p.add_argument("--tile", type=int, default=512)
    # Tiles of one texture are independent, so they can batch: a 2048 source at --tile 512 is
    # 16 tiles in 4 shape groups, 4 forward passes instead of 16. Off by default. The colour
    # slot's model is attention-based and its activations at 4 tiles a pass outran a 12 GB card
    # by a factor of ~3; with the driver's sysmem fallback on, that ran at 253 s a texture
    # instead of failing (see check_sysmem_fallback). Turn it on for light archs with headroom;
    # it is lowered automatically on a real OOM.
    p.add_argument(
        "--tile-batch",
        type=int,
        default=1,
        help="tiles of one texture per forward pass (default 1: off; try 2-4 for light models)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of Lanczos when a model is missing",
    )
    p.add_argument(
        "--icon-batch",
        type=int,
        default=32,
        help="same-sized icon/UI textures per model batch (1 disables batching)",
    )
    # One texconv invocation per batch of finished textures rather than one per texture. Measured
    # on the 4070 SUPER (scripts/bench_texconv.py): a 512x512 BC7 is 0.32 s alone and 0.06 s each
    # in a call of 64, and the curve is flat past 32. The byte cap keeps a batch of 4x outputs
    # from holding gigabytes in RAM and scratch: big textures go alone, small ones by the dozen.
    p.add_argument(
        "--encode-batch",
        type=int,
        default=32,
        help="textures per texconv invocation (1 disables encode batching)",
    )
    p.add_argument(
        "--encode-mb",
        type=int,
        default=256,
        help="upper bound in MiB of upscaled pixels per encode batch",
    )
    p.add_argument(
        "--encode-workers",
        type=int,
        default=1,
        help="encode batches in flight at once (each is its own texconv process)",
    )
    p.set_defaults(fn=cmd_run)
    # pack, qa and modup write to (or read from) the same mod root as run; one default for all of them.
    p = sub.add_parser("pack")
    p.add_argument("--out", default=OUT_DEFAULT, help="(default: %(default)s)")
    p.add_argument(
        "--penumbra-config",
        default=PENUMBRA_CONFIG_DEFAULT,
        help="Penumbra's plugin config dir, to enable the mods in the Default collection (default: %(default)s; pass '' to skip)",
    )
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("probe", help="check which BC7 encoder will be used (GPU texconv or not)")
    p.set_defaults(fn=lambda a: 0 if check_encoder(a) else 2)
    p = sub.add_parser("qa")
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--family", action="append")
    p.add_argument("--samples", type=int, default=20)
    p.set_defaults(fn=cmd_qa)
    p = sub.add_parser(
        "requeue",
        help="put rows back to 'planned' (failed, skipped, or an older recipe)",
    )
    p.add_argument("--failed", action="store_true")
    p.add_argument("--skipped", action="store_true")
    p.add_argument(
        "--old-recipe",
        action="store_true",
        help="done rows finished before models/registry.json existed",
    )
    p.add_argument(
        "--without-model",
        metavar="NAME",
        help="done rows whose models: note does not mention NAME",
    )
    p.add_argument(
        "--done",
        action="store_true",
        help="done rows matching the filters -- for a PROCESSING change no model name can"
        " detect; requires --family/--role/--path-like",
    )
    p.add_argument("--family", action="append")
    p.add_argument("--role", action="append")
    p.add_argument("--path-like", help="SQL LIKE on the path, e.g. chara/equipment/e08%%")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_requeue)
    p = sub.add_parser(
        "reclassify",
        help="re-run classify() over existing rows after a classifier change",
    )
    p.add_argument("--family", action="append")
    p.add_argument("--path-like")
    p.add_argument(
        "--sync-status",
        action="store_true",
        help="also mark planned rows 'skipped' when no role function will ever touch "
        "them (role id/skip/other, non-2D, or a family with no tier), with the "
        "reason in `note` -- the retrofit for rows enumerated before that rule",
    )
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_reclassify)
    p = sub.add_parser(
        "audit", help="report the texture classes that are special-cased or unsupported"
    )
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser(
        "fingerprint",
        help="record/compare a content hash of each source texture, so a patch"
        " cannot silently leave an upscale behind",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="compare against the stored hashes instead of writing",
    )
    p.add_argument(
        "--requeue",
        action="store_true",
        help="with --check: set changed rows back to 'planned'",
    )
    p.add_argument(
        "--restamp",
        action="store_true",
        help="re-take hashes that already exist (adopt the current install)",
    )
    p.add_argument("--all", action="store_true", help="every row, not just status='done'")
    p.add_argument("--family", action="append")
    p.add_argument("--path-like")
    p.add_argument("--log", action="store_true", help="show the recorded fingerprint runs and stop")
    p.add_argument(
        "--export",
        nargs="?",
        const="-",
        metavar="FILE",
        help="write the stamped rows as TSV (default: hash-manifests\\texture-fingerprints.tsv) and stop",
    )
    p.set_defaults(fn=cmd_fingerprint)
    p = sub.add_parser(
        "where",
        help="print the resolved layout (database, models, texconv, KB tools, path list)",
    )
    p.set_defaults(fn=lambda _a: print(paths.describe()) or 0)
    p = sub.add_parser("modup", help="upscale the textures inside existing mods (icon packs)")
    p.add_argument("--mod", action="append", required=True)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--models", default=paths.MODELS)
    p.add_argument("--device")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--icon-scale", type=int, default=4)
    p.add_argument("--sheet-scale", type=int, default=2)
    p.add_argument("--penumbra-config")
    p.set_defaults(fn=cmd_modup)
    a = ap.parse_args(argv)
    # Progress and results go to stdout with print(); anything a user would only want while
    # chasing a problem goes through logging, which --verbose turns on.
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
