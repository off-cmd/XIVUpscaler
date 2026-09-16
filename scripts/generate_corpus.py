"""Generate a deterministic stratified benchmark corpus from manifest.sqlite.

Run once to create corpus.json. The selected paths are frozen so future benchmark
runs always test the exact same textures for apples-to-apples comparison.
"""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clarity import paths

BENCHMARK_DIR = os.path.join(paths.TOOL_ROOT, "benchmarks", "color")
CORPUS_FILE = os.path.join(BENCHMARK_DIR, "corpus.json")

conn = sqlite3.connect(paths.DB)
cur = conn.cursor()

# Stratified categories: family prefix × source format.
# We want diversity across surface types (metal, cloth, stone, organic, painted, smooth).
# Use deterministic ordering (path ASC) with LIMIT+OFFSET to get reproducible variety
# rather than RANDOM() which changes every run.
strata = [
    # Character gear — almost entirely BC1 in FFXIV
    ("equipment_BC1", "family='equipment'  AND fmt='BC1'"),
    ("accessory_BC1", "family='accessory'  AND fmt='BC1'"),
    ("weapon_BC1", "family='weapon'     AND fmt='BC1'"),
    # Monsters — mix of BC1 and BC7
    ("monster_BC1", "family='monster'    AND fmt='BC1'"),
    ("monster_BC7", "family='monster'    AND fmt='BC7'"),
    # Demihumans
    ("demihuman_BC1", "family='demihuman'  AND fmt='BC1'"),
    # Human body/face (skin textures — these go through SkinDiffDDS cleaning)
    ("human-body_BC1", "family='human-body' AND fmt='BC1'"),
    ("human-face_BC1", "family='human-face' AND fmt='BC1'"),
    # Environments — the biggest visual variety
    ("bg_BC1", "family LIKE 'bg%'   AND fmt='BC1'"),
    ("bg_BC7", "family LIKE 'bg%'   AND fmt='BC7'"),
    # Housing interiors (viewed up close)
    (
        "housing_BC1",
        "(family='bg-hou' OR family='bg-ind' OR family='bgcommon-hou') AND fmt='BC1'",
    ),
    (
        "housing_BC7",
        "(family='bg-hou' OR family='bg-ind' OR family='bgcommon-hou') AND fmt='BC7'",
    ),
]

PER_STRATUM = 4
corpus = {}

for name, where in strata:
    # Deterministic spread: order by path, take every Nth row to get variety across
    # the entire alphabet of asset IDs rather than clustering on one.
    cur.execute(f"""
        SELECT path FROM tex
        WHERE role='color' AND w >= 256 AND w <= 2048 AND ({where})
        ORDER BY path
    """)
    all_paths = [r[0] for r in cur.fetchall()]

    if len(all_paths) <= PER_STRATUM:
        selected = all_paths
    else:
        # Evenly space selections across the full sorted list
        step = len(all_paths) / PER_STRATUM
        selected = [all_paths[int(i * step)] for i in range(PER_STRATUM)]

    corpus[name] = selected

conn.close()

os.makedirs(BENCHMARK_DIR, exist_ok=True)
with open(CORPUS_FILE, "w") as f:
    json.dump(corpus, f, indent=2)

total = sum(len(v) for v in corpus.values())
print(f"Corpus generated: {total} textures across {len(corpus)} strata.")
for name, paths_list in corpus.items():
    print(f"  {name}: {len(paths_list)}")
