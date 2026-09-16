"""clarity-upscale — the G4 pipeline.

plan   enumerate every vanilla texture, classify it (family × role), store in manifest.sqlite
run    upscale at the top tier per role (GPU, resumable, --budget seconds), derive the lower tiers
pack   write Penumbra mod folders with a "Tier" option group per family
qa     contamination / normal-length / colour-drift checks and side-by-side sheets
"""
