import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spandrel import ModelLoader

from clarity import paths
from clarity.ffxiv import sqpack

BENCHMARK_DIR = os.path.join(paths.TOOL_ROOT, "benchmarks", "color")
CORPUS_FILE = os.path.join(BENCHMARK_DIR, "corpus.json")
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "results")
CROP_SIZE = 512  # Size of the 1:1 crop on the output (upscaled) image

# Auto-detect FFXIV sqpack directory. Override with FFXIV_SQPACK env var.
_SQPACK_CANDIDATES = [
    r"C:\Program Files (x86)\Steam\steamapps\common\FINAL FANTASY XIV Online\game\sqpack",
    r"C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game\sqpack",
]


def _find_sqpack():
    env = os.environ.get("FFXIV_SQPACK")
    if env and os.path.isdir(env):
        return env
    for p in _SQPACK_CANDIDATES:
        if os.path.isdir(p):
            return p
    return None


FFXIV_DIR = _find_sqpack()
# ---------------------


def load_texture_from_sqpack(path, _):
    from clarity import texio

    try:
        hdr, rgba = texio.read(path)
        return rgba, hdr.format_name
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None


def numpy_to_tensor(img):
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    img = (tensor.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(img)


def run_model(model, tensor, eng):
    with torch.no_grad():
        return eng._tiled(model, tensor, model.scale)


def crop_center(img, size):
    width, height = img.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return img.crop((left, top, right, bottom))


def main():
    if FFXIV_DIR is None:
        print("Error: Could not find FFXIV sqpack directory.")
        print("Set the FFXIV_SQPACK environment variable to your game's sqpack folder.")
        return

    if not os.path.exists(CORPUS_FILE):
        print(f"Error: Corpus not found at {CORPUS_FILE}")
        return

    with open(CORPUS_FILE) as f:
        corpus = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = ModelLoader()
    models_dir = paths.MODELS

    print(f"Loading FFXIV sqpack indices from {FFXIV_DIR}...")
    game = sqpack.GameData(FFXIV_DIR)

    # Hand the game instance to clarity.ffxiv so texio.read() resolves paths without
    # re-locating the install.
    from clarity import ffxiv as kb

    kb._GD = game

    # Load models. (Spandrel ESRGAN models are typically ~60MB each, so holding 3 in 12GB VRAM is trivial)
    print("Loading models into VRAM...")
    rplksrd = (
        loader.load_from_file(os.path.join(models_dir, "4x-PBRify_RPLKSRd_V3.pth"))
        .eval()
        .to(device)
    )
    upscalerv4 = (
        loader.load_from_file(os.path.join(models_dir, "4x-PBRify_UpscalerV4.pth"))
        .eval()
        .to(device)
    )
    bc1_cleaner = (
        loader.load_from_file(os.path.join(models_dir, "1x_BC1-smooth2.pth")).eval().to(device)
    )

    from clarity.processing import engine

    eng = engine.Engine(models_dir, device=str(device))

    for category, paths_list in corpus.items():
        print(f"\n--- Processing Category: {category} ---")
        for idx, path in enumerate(paths_list):
            basename = path.replace("/", "_").replace(".tex", "")
            print(f"[{idx + 1}/{len(paths_list)}] {basename}...")

            rgba, fmt = load_texture_from_sqpack(path, game)
            if rgba is None:
                print(f"  Failed to load {path}")
                continue

            rgb = rgba[..., :3]
            orig_img = Image.fromarray(rgb)
            tensor_rgb = numpy_to_tensor(rgb).to(device)

            # Outputs dictionary: { "name": PIL.Image }
            outputs = {"1_vanilla": orig_img}

            # Direct upscales
            outputs["2_RPLKSRd_V3"] = tensor_to_pil(run_model(rplksrd, tensor_rgb, eng))
            outputs["3_UpscalerV4"] = tensor_to_pil(run_model(upscalerv4, tensor_rgb, eng))

            # BC1 cleaning combinations (ONLY for BC1 sources)
            if fmt == "BC1":
                cleaned = run_model(bc1_cleaner, tensor_rgb, eng)
                outputs["4_BC1smooth2_only"] = tensor_to_pil(cleaned)
                outputs["5_BC1smooth2_RPLKSRd"] = tensor_to_pil(run_model(rplksrd, cleaned, eng))
                outputs["6_BC1smooth2_UpscalerV4"] = tensor_to_pil(
                    run_model(upscalerv4, cleaned, eng)
                )

            # Save results and 1:1 crops
            out_folder = os.path.join(OUTPUT_DIR, category, basename)
            os.makedirs(out_folder, exist_ok=True)

            for name, img in outputs.items():
                # Save full image
                img.save(os.path.join(out_folder, f"{name}.png"))

                # Save 1:1 center crop (scale crop size for native/1x images)
                is_native = name in ("1_vanilla", "4_BC1smooth2_only")
                crop_s = CROP_SIZE // 4 if is_native else CROP_SIZE

                # Only crop if the image is actually larger than the crop size
                if img.width >= crop_s and img.height >= crop_s:
                    crop = crop_center(img, crop_s)

                    # For vanilla, nearest-neighbor scale it up 4x so it matches the upscale crops visually!
                    if is_native:
                        crop = crop.resize(
                            (CROP_SIZE, CROP_SIZE), resample=Image.Resampling.NEAREST
                        )

                    crop.save(os.path.join(out_folder, f"{name}_crop.png"))

            # Save metadata
            meta = {
                "path": path,
                "category": category,
                "format": fmt,
                "resolution": f"{orig_img.width}x{orig_img.height}",
            }
            with open(os.path.join(out_folder, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
