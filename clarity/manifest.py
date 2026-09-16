"""The manifest: every vanilla texture we intend to touch, classified family × role, in sqlite.

Two ways to enumerate, used together:

* **generative** — walk the game's own structure: equipment / accessory / weapon / monster /
  demihuman / human sets by id, `.mdl` material names × `.imc` material variants → `.mtrl` →
  texture paths; UI icons by id probing.  Complete for `chara/` and `ui/icon/` without any
  external list.
* **path list** — a ResLogger `CurrentPathList` (or any text file with one game path per line)
  covers `bg/`, `bgcommon/`, `ui/uld/` and everything else.  `clarity plan --pathlist FILE`.

Classification (KB 07 §5, KB 08):

    role      suffix                      what the pipeline does
    id        _id / _index                nothing — discrete colour-set selector, never resampled
    normal    _n / _norm                  BC1-clean → RG normal model → renormalise; alpha (opacity) as a scalar
    mask      _m / _mask                  each channel as an independent scalar
    color     _d / _s / _b / _base / _e   colour model + average-colour fix
    icon      ui/icon/**                  RGBA, alpha-aware, pixel-art model
    ui        ui/uld/**, ui/other         RGBA, alpha-aware
    skip      flow maps, arrays, cubes, depth, float formats, vfx
"""

import re
import sqlite3
import time

from . import ffxiv as kb

FAMILIES = [
    ("chara/equipment/", "equipment"),
    ("chara/accessory/", "accessory"),
    ("chara/weapon/", "weapon"),
    ("chara/monster/", "monster"),
    ("chara/demihuman/", "demihuman"),
    ("chara/human/", "human"),
    ("bgcommon/", "bgcommon"),
    ("bg/", "bg"),
    ("ui/icon/", "ui-icon"),
    ("ui/uld/", "ui-uld"),
    ("ui/", "ui-other"),
    ("vfx/", "vfx"),
    ("common/", "common"),
]
HUMAN_PART = re.compile(r"chara/human/c\d{4}/obj/(body|face|hair|tail|zear)/")

# HOUSING AND INTERIORS ARE SPLIT OUT, BUT NOT ACROSS THE ARCHIVE BOUNDARY.
#
# The useful distinction is the one Kartoffels draws: a wall you stand two feet from in your own
# house is looked at for hours, and a cliff face three hundred yalms away is not. But his categories
# straddle `bg/` and `bgcommon/`, and those are SEPARATE SQPACK ARCHIVES -- category 0x02 and 0x01,
# different .dat/.index files, different repos -- so merging them into one family would put paths
# from two archives behind one policy and one mod, and any future per-archive operation would have
# to unpick it. So the split happens INSIDE each prefix:
#
#   bg/<repo>/<zone>/<kind>/...     kind is segment 3: fld dun twn btl evt ang pvp ... and hou / ind
#   bgcommon/<group>/<sub>/...      group is segment 1: hou (indoor outdoor craft dyna common),
#                                   world (cut evt air sip emp itm tbx ...), mji, nature ...
#
# The families carry what POLICY and the Penumbra mod list need; the finer level goes in `part`,
# which the schema already has and which nothing was putting anything in for these two prefixes.
# That is why bgcommon-hou is one family rather than four: indoor/outdoor/craft/dyna want the same
# tier as each other, and `part` already distinguishes them for querying and for --family filtering.
BG_HOUSING_KINDS = {"hou", "ind"}
BGCOMMON_GROUPS = {"hou", "mji"}
SUFFIX_ROLE = {
    "id": "id",
    "index": "id",
    "n": "normal",
    "norm": "normal",
    "m": "mask",
    "mask": "mask",
    "d": "color",
    "s": "color",
    "b": "color",
    "base": "color",
    "e": "color",
    "emissive": "color",
    "a": "color",
    "diffuse": "color",
    "spec": "color",
    "specular": "color",
    "c": "color",
    "flow": "skip",
    "flw": "skip",
}
SKIP_FORMATS = {
    "R32F",
    "R16G16F",
    "R32G32F",
    "R16G16B16A16F",
    "R32G32B32A32F",
    "D16",
    "D24S8",
    "Null",
    "Shadow16",
    "Shadow24",
    "BC6H",
    "BC4",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tex (
    path TEXT PRIMARY KEY, family TEXT, part TEXT, role TEXT,
    w INTEGER, h INTEGER, fmt TEXT, mips INTEGER, bytes INTEGER, ttype TEXT,
    status TEXT DEFAULT 'planned', tiers TEXT DEFAULT '', note TEXT DEFAULT '', updated REAL
);
CREATE INDEX IF NOT EXISTS tex_family ON tex(family, role, status);

-- One row per `clarity fingerprint` invocation: which patch the install was on, and what the pass
-- found. This is the audit trail that makes "did 7.6 touch anything we upscaled?" answerable months
-- later, when nobody remembers which build the mods were made from.
CREATE TABLE IF NOT EXISTS fpsnap (
    ts REAL, version TEXT, mode TEXT,
    n_hashed INTEGER, n_changed INTEGER, n_gone INTEGER, n_new INTEGER, note TEXT DEFAULT ''
);
"""

# Columns added after the first schema shipped. sqlite has no `ADD COLUMN IF NOT EXISTS`, and a
# manifest that predates a column is the normal case rather than the exception, so each one is
# attempted and its "duplicate column name" is the success path.
MIGRATIONS = [
    "ALTER TABLE tex ADD COLUMN recipe TEXT DEFAULT ''",
    # The content hash of the source .tex as it stood when this row was last processed. Empty means
    # "never fingerprinted", which is different from "unchanged" -- `fingerprint --check` says so.
    "ALTER TABLE tex ADD COLUMN srchash TEXT DEFAULT ''",
    # The game version the hash was taken on, so a stale hash can be attributed to a patch.
    "ALTER TABLE tex ADD COLUMN srcver TEXT DEFAULT ''",
]


def classify(path, hdr):
    """-> (family, part, role) for a game path with a parsed TexHeader (or None)."""
    family = "other"
    for prefix, fam in FAMILIES:
        if path.startswith(prefix):
            family = fam
            break
    part = ""
    if family == "human":
        m = HUMAN_PART.match(path)
        part = m.group(1) if m else ""
        family = "human-" + (part or "other")
    elif family == "bg":
        seg = path.split("/")
        kind = seg[3] if len(seg) > 3 else ""
        if kind in BG_HOUSING_KINDS:
            family = "bg-" + kind
            part = seg[4] if len(seg) > 4 else ""  # dyna / indoor / the ward's own sub-kind
        else:
            part = kind  # fld / dun / twn / btl / evt ...
    elif family == "bgcommon":
        seg = path.split("/")
        group = seg[1] if len(seg) > 1 else ""
        part = seg[2] if len(seg) > 2 else ""  # indoor / outdoor / craft / dyna / cut ...
        if group in BGCOMMON_GROUPS:
            family = "bgcommon-" + group
    name = path.rsplit("/", 1)[-1][:-4]
    suffix = name.rsplit("_", 1)[-1] if "_" in name else ""
    if family == "ui-icon":
        role = "icon"
    elif family in ("ui-uld", "ui-other"):
        role = "ui"
    elif family == "vfx":
        role = "skip"
    elif family == "common" and name.startswith("-nowloading"):
        # The loading screens, and nothing else in `common`.
        #
        # 1920x1080 BC1 painted art shown full-screen, so at 2x it lands on exactly 3840x2160 and
        # you look at it every zone change. `ui` rather than `color` follows the rule a few lines
        # down that already sends big art in ui/icon to UltraSharpV2 -- these are the same kind of
        # picture, just filed elsewhere.
        #
        # Deliberately one prefix and not the family. `common` also holds seven font atlases, the
        # software cursor, and a pile of shader lookup tables (-attenuation, -dither, -fresnel,
        # -noise, -caustics, -line00*, -sss). Those are data sampled by shaders at exact
        # coordinates, not pictures: upscaling a font atlas moves every glyph out from under the
        # UVs that address it. They stay role `other` and PROCESSED_ROLES never touches them, so
        # giving the family a tier cannot reach them.
        role = "ui"
    else:
        # World textures default to `color`: bg/ and bgcommon/ name a great many maps with suffixes
        # that are not in SUFFIX_ROLE at all, and they are almost all albedo. `startswith` rather
        # than an equality test, so the housing sub-families inherit it -- the first version of the
        # split used `in ("bg", "bgcommon")` and quietly sent every bg-hou texture to role `other`,
        # which PROCESSED_ROLES never touches. A family split must not change what a role is.
        role = SUFFIX_ROLE.get(
            suffix, "color" if family.startswith(("bg", "bgcommon")) else "other"
        )
    if hdr is not None:
        if hdr.texture_type != "2D" or hdr.format_name in SKIP_FORMATS or hdr.depth > 1:
            role = "skip"
        if role == "icon" and hdr.width > 512:  # ui/icon also holds big art (maps, loading)
            role = "ui"
    return family, part, role


class StoredHeader:
    """The subset of a TexHeader that classify() actually reads, rebuilt from manifest columns.

    `reclassify` has to re-run the classifier over rows whose .tex it is not going to open again.
    Passing None for the header made it re-run only the path half, which reverses every rule that
    consults the texture. The manifest stores w, h, fmt and ttype, and those are exactly the four
    fields classify() looks at, so a row can be re-classified offline with the same verdict the
    original scan reached.
    """

    __slots__ = ("depth", "format_name", "height", "texture_type", "width")

    def __init__(self, w, h, fmt, ttype):
        self.width, self.height = w or 0, h or 0
        self.format_name, self.texture_type = fmt or "", ttype or "2D"
        self.depth = 1  # a depth > 1 row was already turned into role 'skip' at scan time


# The roles that have a function in roles.ROLE_FN. Kept here rather than in cli so that enumeration
# can consult it, which is the whole point of skip_reason below.
PROCESSED_ROLES = ("normal", "mask", "color", "icon", "ui")

_NO_PATH = {
    "id": "index map: R is a colour-set row in multiples of 17 and G a blend weight, so any "
    "resampling corrupts dye behaviour",
    "skip": "not a 2D colour texture the role functions can operate on (array, cube, depth, float "
    "format, or vfx)",
    "other": "no suffix rule maps this name to a role (font atlas, shader lookup table, _catc, "
    "_mult, and similar)",
}


def skip_reason(family, role, ttype=None):
    """Why this row will never be processed, or None if it will be.

    QUEUE DEPTH HAS TO MEAN SOMETHING. Every row landed as status='planned', but `run` only ever
    walks the five roles in PROCESSED_ROLES, so 24,401 rows -- id 21,015, skip 3,069, other 317 --
    sat in the queue forever looking like outstanding work. Nothing was broken by it (the engine
    never sees them, which is why only a handful of rows have ever failed), but "planned" was
    quietly being used to mean two different things, and the skip policy lived in the shape of a
    loop rather than anywhere a person could audit it.

    This is that policy, written once, consulted at enumeration time and by `reclassify
    --sync-status` for rows that predate it. The reason string goes in `note`, so a row that will
    never be processed says so and says why.
    """
    if role not in PROCESSED_ROLES:
        return _NO_PATH.get(role, f"role {role!r} has no processing function")
    if ttype and ttype != "2D":
        # Belt and braces: classify() already turns these into role 'skip' when it has a header,
        # but a row inserted before that rule existed would not have been.
        return f"texture type {ttype}: the role functions are 2D operations"
    from .processing import roles as _roles

    if _roles.POLICY.get(family, (None, 0))[0] is None:
        return f"family {family!r} has no tier in roles.POLICY"
    return None


def sync_skipped(man, dry_run=False):
    """Set status='skipped' on planned rows that skip_reason() says will never be processed.

    -> (n_changed, {reason: count}). Only touches status='planned': a row already done or failed
    describes something that actually happened and is not this function's business.
    """
    import collections

    changes, why = [], collections.Counter()
    for path, family, role, ttype in man.db.execute(
        "SELECT path, family, role, ttype FROM tex WHERE status='planned'"
    ):
        r = skip_reason(family, role, ttype)
        if r:
            changes.append((r[:400], path))
            why[r.split(":")[0]] += 1
    if changes and not dry_run:
        man.db.executemany("UPDATE tex SET status='skipped', note=? WHERE path=?", changes)
        man.db.commit()
    return len(changes), why


class Manifest:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        for stmt in MIGRATIONS:
            try:
                self.db.execute(stmt)
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise
        self.db.commit()

    def add(self, path, family, part, role, hdr):
        # A row with no processing path is born 'skipped' with the reason in `note`, not 'planned'.
        # Writing the verdict at insert time is what keeps the queue honest going forward;
        # `reclassify --sync-status` is the retrofit for everything inserted before this existed.
        why = skip_reason(family, role, hdr.texture_type)
        self.db.execute(
            "INSERT OR IGNORE INTO tex(path,family,part,role,w,h,fmt,mips,bytes,ttype,status,note,updated)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                path,
                family,
                part,
                role,
                hdr.width,
                hdr.height,
                hdr.format_name,
                hdr.mip_count,
                hdr.total_size(),
                hdr.texture_type,
                "skipped" if why else "planned",
                (why or "")[:400],
                time.time(),
            ),
        )

    def commit(self):
        self.db.commit()

    def count(self, **where):
        q = "SELECT COUNT(*) FROM tex"
        args = []
        if where:
            q += " WHERE " + " AND ".join(f"{k}=?" for k in where)
            args = list(where.values())
        return self.db.execute(q, args).fetchone()[0]

    def rows(self, family=None, role=None, status=None, limit=None, since=None):
        q, args = (
            "SELECT path,family,part,role,w,h,fmt,mips,status,tiers FROM tex WHERE 1=1",
            [],
        )
        if family:
            q += " AND family=?"
            args.append(family)
        if role:
            q += " AND role=?"
            args.append(role)
        if status:
            q += " AND status=?"
            args.append(status)
        if since:
            # The patch delta: rows a re-enumeration added on or after `since` (their `updated` is
            # the insert time) plus rows `fingerprint --check --requeue` sent back (their note
            # names the version). Everything the previous patch left unfinished stays out of it.
            q += " AND (updated >= ? OR note LIKE 'changed at %')"
            args.append(since)
        q += " ORDER BY path"
        if limit:
            q += " LIMIT %d" % limit
        return self.db.execute(q, args).fetchall()

    def set_status(self, path, status, tiers="", note="", srchash=None, srcver=""):
        """srchash=None leaves whatever hash the row already had; pass a hash when the row is being

        marked done so the stamp and the output are written in the same transaction and can never
        disagree about which source the files on disk came from.
        """
        if srchash is None:
            self.db.execute(
                "UPDATE tex SET status=?, tiers=?, note=?, updated=? WHERE path=?",
                (status, tiers, note, time.time(), path),
            )
        else:
            self.db.execute(
                "UPDATE tex SET status=?, tiers=?, note=?, updated=?, srchash=?, srcver=? WHERE path=?",
                (status, tiers, note, time.time(), srchash, srcver, path),
            )

    def set_hash(self, path, srchash, srcver):
        self.db.execute("UPDATE tex SET srchash=?, srcver=? WHERE path=?", (srchash, srcver, path))

    def snapshot(self, version, mode, n_hashed, n_changed, n_gone, n_new, note=""):
        self.db.execute(
            "INSERT INTO fpsnap(ts,version,mode,n_hashed,n_changed,n_gone,n_new,note)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), version, mode, n_hashed, n_changed, n_gone, n_new, note),
        )
        self.db.commit()

    def snapshots(self, limit=20):
        return self.db.execute(
            "SELECT ts,version,mode,n_hashed,n_changed,n_gone,n_new,note"
            " FROM fpsnap ORDER BY ts DESC LIMIT %d" % limit
        ).fetchall()

    def summary(self):
        return self.db.execute(
            "SELECT family, role, status, COUNT(*), SUM(bytes) FROM tex GROUP BY family, role, status ORDER BY family, role, status"
        ).fetchall()


# ---------------------------------------------------------------- enumeration
def _header(gd, path):
    loc = gd.locate(path)
    if not loc:
        return None
    raw = kb.sqpack.read_tex_header(*loc)
    return kb.texfile.TexHeader(raw) if raw else None


def _mtrl_textures(gd, mtrl_path):
    """Texture paths named by a material. kbtools' class is Mtrl (an earlier name here, MtrlFile,

    raised AttributeError on every call, so the structure walk had never added a texture).
    """
    try:
        m = kb.mtrlfile.Mtrl(gd.read(mtrl_path))
    except Exception:
        return []
    return [t for t, _ in m.textures if t.endswith(".tex")]


def _model_materials(gd, mdl_path):
    try:
        return kb.mdlstrings.read(gd.read(mdl_path))[1]["material"]
    except Exception:
        return []


RACES = [
    "0101",
    "0104",
    "0201",
    "0204",
    "0301",
    "0304",
    "0401",
    "0404",
    "0501",
    "0504",
    "0601",
    "0604",
    "0701",
    "0704",
    "0801",
    "0804",
    "0901",
    "0904",
    "1001",
    "1004",
    "1101",
    "1104",
    "1201",
    "1204",
    "1301",
    "1304",
    "1401",
    "1404",
    "1501",
    "1504",
    "1601",
    "1604",
    "1701",
    "1704",
    "1801",
    "1804",
]
EQ_SLOTS = ["met", "top", "glv", "dwn", "sho"]
AC_SLOTS = ["ear", "nek", "wrs", "rir", "ril"]


def _variants(gd, imc_path):
    """Material ids used by any variant of an .imc (default 1 when the imc is absent)."""
    if not gd.exists(imc_path):
        return {1}
    try:
        imc = kb.imcfile.ImcFile(gd.read(imc_path))
    except Exception:
        return {1}
    ids = set()
    for parts in imc.variants:
        for e in parts:
            if e.material_id:
                ids.add(e.material_id)
    return ids or {1}


def _add_paths(man, gd, paths, seen, stats):
    for tp in paths:
        if tp in seen:
            continue
        seen.add(tp)
        hdr = _header(gd, tp)
        if hdr is None:
            stats["missing"] += 1
            continue
        fam, part, role = classify(tp, hdr)
        man.add(tp, fam, part, role, hdr)
        stats["added"] += 1


def gen_sets(man, gd, seen, stats, kind, ids, slots, model_fmt, imc_fmt, mat_dir_fmt, log):
    """Equipment / accessory style sets: per set id, per race model → materials × imc variants."""
    t0 = time.time()
    found = 0
    for sid in ids:
        mats = {}
        for r in RACES:
            for slot in slots:
                mp = model_fmt.format(kind=kind, sid=sid, r=r, slot=slot)
                if gd.exists(mp):
                    for name in _model_materials(gd, mp):
                        mats[name] = True
        if not mats:
            continue
        found += 1
        vids = _variants(gd, imc_fmt.format(kind=kind, sid=sid))
        paths = []
        for name in mats:
            for v in vids:
                if name.startswith("/"):
                    paths.append(mat_dir_fmt.format(kind=kind, sid=sid, v=v) + name[1:])
                else:
                    paths.append(name)
        # every material that exists, then its textures
        texs = []
        for mp in paths:
            if gd.exists(mp):
                texs.extend(_mtrl_textures(gd, mp))
        _add_paths(man, gd, texs, seen, stats)
    log("%s: %d sets, %.0fs" % (kind, found, time.time() - t0))


def gen_chara(man, gd, log=print):
    seen, stats = set(), {"added": 0, "missing": 0}
    gen_sets(
        man,
        gd,
        seen,
        stats,
        "equipment",
        range(1, 10000),
        EQ_SLOTS,
        "chara/{kind}/e{sid:04d}/model/c{r}e{sid:04d}_{slot}.mdl",
        "chara/{kind}/e{sid:04d}/e{sid:04d}.imc",
        "chara/{kind}/e{sid:04d}/material/v{v:04d}/",
        log,
    )
    gen_sets(
        man,
        gd,
        seen,
        stats,
        "accessory",
        range(1, 10000),
        AC_SLOTS,
        "chara/{kind}/a{sid:04d}/model/c{r}a{sid:04d}_{slot}.mdl",
        "chara/{kind}/a{sid:04d}/a{sid:04d}.imc",
        "chara/{kind}/a{sid:04d}/material/v{v:04d}/",
        log,
    )
    # weapons / monsters: obj/body/bNNNN
    #
    # THE BODY SCAN USED TO BE `range(1, 60)` WITH `if b > 2: break`, AND BOTH HALVES WERE WRONG.
    #
    # The cap excluded every body numbered 60 or above outright -- w1851 runs to b0117 in the
    # 2026.09.01 client and w9001 to b0468 -- and the break stopped the scan at the first gap, so a
    # set whose numbering is not contiguous from b0001 was truncated at its first hole. w0371 has
    # bodies 2, 3, 8, 9, 13, 14, 16, 19, 40: the loop saw b0002 and b0003, hit the gap at b0004 and
    # stopped. w2651's lowest body is b0011, so the loop hit `b > 2` at b0003 and produced NOTHING
    # for that set at all.
    #
    # The path list papered over most of it -- ResLogger sees whatever anyone loaded -- which is why
    # this only surfaced as nine weapon models missing from a 46,451-path comparison. It was never
    # about those nine models' material shape (the "no _d" theory does not survive contact with the
    # data: n/m/id is the MOST COMMON shape in the manifest, 2,102 of 4,913 weapon model dirs).
    #
    # So: probe a short prefix to decide whether the set exists at all, then scan the set's whole
    # range with no early exit. PROBE has to clear the highest first-body in the client, which is
    # b0011 (one weapon set); 24 leaves room. SCAN covers b0468. The lone w2701 b9998 is left out
    # deliberately -- scanning to 10,000 for one path costs 10 million index lookups.
    BODY_SCAN = {"weapon": (24, 512), "monster": (8, 64)}
    for kind, letter, top in (("weapon", "w", 10000), ("monster", "m", 10000)):
        t0, found = time.time(), 0
        probe, scan = BODY_SCAN[kind]
        mdl = f"chara/{kind}/{letter}%04d/obj/body/b%04d/model/{letter}%04db%04d.mdl"
        for sid in range(1, top):
            if not any(gd.exists(mdl % (sid, b, sid, b)) for b in range(1, probe + 1)):
                continue
            for b in range(1, scan + 1):
                mp = mdl % (sid, b, sid, b)
                if not gd.exists(mp):
                    continue
                found += 1
                vids = _variants(
                    gd,
                    "chara/%s/%s%04d/obj/body/b%04d/b%04d.imc" % (kind, letter, sid, b, b),
                )
                texs = []
                for name in _model_materials(gd, mp):
                    for v in vids:
                        p = (
                            (
                                "chara/%s/%s%04d/obj/body/b%04d/material/v%04d/"
                                % (kind, letter, sid, b, v)
                                + name[1:]
                            )
                            if name.startswith("/")
                            else name
                        )
                        if gd.exists(p):
                            texs.extend(_mtrl_textures(gd, p))
                _add_paths(man, gd, texs, seen, stats)
        log("%s: %d bodies, %.0fs" % (kind, found, time.time() - t0))
    # demihuman: obj/equipment/eNNNN  # noqa: ERA001 - a path pattern, not code
    t0, found = time.time(), 0
    for d in range(1, 3000):
        base = "chara/demihuman/d%04d/obj/equipment/" % d
        if not any(
            gd.exists(base + "e%04d/model/d%04de%04d_%s.mdl" % (e, d, e, s))
            for e in (1, 2)
            for s in EQ_SLOTS
        ):
            continue
        for e in range(1, 400):
            texs = []
            for s in EQ_SLOTS:
                mp = base + "e%04d/model/d%04de%04d_%s.mdl" % (e, d, e, s)
                if not gd.exists(mp):
                    continue
                found += 1
                vids = _variants(gd, base + "e%04d/e%04d.imc" % (e, e))
                for name in _model_materials(gd, mp):
                    for v in vids:
                        p = (
                            (base + "e%04d/material/v%04d/" % (e, v) + name[1:])
                            if name.startswith("/")
                            else name
                        )
                        if gd.exists(p):
                            texs.extend(_mtrl_textures(gd, p))
            _add_paths(man, gd, texs, seen, stats)
    log("demihuman: %d models, %.0fs" % (found, time.time() - t0))
    # human parts
    t0, found = time.time(), 0
    # The `top` per part is a scan bound, and two of them were below what the client ships: body
    # reaches b0201 and tail t0202 (the 02xx block), against caps of 20 and 40. Both were only in
    # the manifest at all because the path list happened to carry them. 300 covers every part in the
    # 2026.09.01 client with room; face and hair were already fine at 400 (they reach 251 and 249).
    for r in RACES:
        for part, letter, top, slots in (
            ("body", "b", 300, ["top", "dwn", "glv", "sho"]),
            ("face", "f", 400, ["fac"]),
            ("hair", "h", 400, ["hir"]),
            ("tail", "t", 300, ["til"]),
            ("zear", "z", 40, ["zer"]),
        ):
            for n in range(1, top):
                mats, any_model = {}, False
                for slot in slots:
                    mp = "chara/human/c%s/obj/%s/%s%04d/model/c%s%s%04d_%s.mdl" % (
                        r,
                        part,
                        letter,
                        n,
                        r,
                        letter,
                        n,
                        slot,
                    )
                    if gd.exists(mp):
                        any_model = True
                        for name in _model_materials(gd, mp):
                            mats[name] = True
                if not any_model:
                    continue
                found += 1
                texs = []
                for name in mats:
                    cands = []
                    if name.startswith("/"):
                        base = "chara/human/c%s/obj/%s/%s%04d/material/" % (
                            r,
                            part,
                            letter,
                            n,
                        )
                        cands = [base + name[1:]] + [
                            base + "v%04d/" % v + name[1:] for v in range(1, 6)
                        ]
                    else:
                        cands = [name]
                    for p in cands:
                        if gd.exists(p):
                            texs.extend(_mtrl_textures(gd, p))
                _add_paths(man, gd, texs, seen, stats)
    log("human: %d parts, %.0fs" % (found, time.time() - t0))
    man.commit()
    return stats


def gen_icons(man, gd, log=print, top=300000):
    seen, stats = set(), {"added": 0, "missing": 0}
    t0 = time.time()
    for i in range(0, top):
        folder = "ui/icon/%06d/" % (i // 1000 * 1000)
        for suffix in ("", "_hr1"):
            for lang in ("", "en/", "ja/", "de/", "fr/"):
                p = folder + lang + "%06d%s.tex" % (i, suffix)
                if gd.locate(p):
                    _add_paths(man, gd, [p], seen, stats)
    man.commit()
    log("icons: %d, %.0fs" % (stats["added"], time.time() - t0))
    return stats


def from_pathlist(
    man,
    gd,
    file,
    log=print,
    only_prefixes=("bg/", "bgcommon/", "ui/", "chara/", "vfx/", "common/"),
):
    seen, stats = set(), {"added": 0, "missing": 0}
    t0 = time.time()
    n = 0
    opener = open
    if file.endswith(".gz"):
        import gzip

        opener = gzip.open
    with opener(file, "rt", encoding="utf-8", errors="ignore") as f:
        batch = []
        for line in f:
            p = line.strip().rsplit(",", 1)[-1].strip().lower()
            if not p.endswith(".tex") or not p.startswith(only_prefixes):
                continue
            batch.append(p)
            n += 1
            if len(batch) >= 5000:
                _add_paths(man, gd, batch, seen, stats)
                batch = []
                man.commit()
        _add_paths(man, gd, batch, seen, stats)
    man.commit()
    log(
        "pathlist: %d .tex lines, %d added, %d not in the index, %.0fs"
        % (n, stats["added"], stats["missing"], time.time() - t0)
    )
    return stats
