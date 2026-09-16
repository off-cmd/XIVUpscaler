import glob
import json
import os
import sys

from clarity import paths

DEFAULT_RESULTS = os.path.join(paths.TOOL_ROOT, "benchmarks", "color", "results")


def generate_viewer(results_dir=DEFAULT_RESULTS):
    output_html = os.path.join(results_dir, "viewer.html")

    categories = sorted(
        [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    )

    html = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clarity A/B Benchmark Viewer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #eee; margin: 0; padding: 20px; padding-bottom: 80px; }
        h1, h2 { color: #fff; }
        .category-section { margin-bottom: 50px; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .specimen { margin-bottom: 40px; background: #2a2a2a; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .specimen-title { font-size: 1.2em; margin-bottom: 15px; color: #61dafb; word-break: break-all; }
        .meta { font-size: 0.9em; color: #aaa; margin-bottom: 15px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .card { background: #222; border: 2px solid #444; border-radius: 4px; padding: 10px; text-align: center; transition: border-color 0.2s; position: relative; }
        .card.selected { border-color: #4caf50; background: #2e3e2e; }
        .card-title { font-size: 0.9em; margin-bottom: 10px; font-weight: bold; }
        .card img { max-width: 100%; height: auto; border: 1px solid #000; cursor: pointer; transition: transform 0.2s; image-rendering: pixelated; }
        .card img:hover { transform: scale(1.02); }
        .tabs { margin-bottom: 10px; }
        .tab-btn { background: #333; color: white; border: none; padding: 5px 15px; cursor: pointer; border-radius: 4px; margin-right: 5px; }
        .tab-btn.active { background: #61dafb; color: #000; }
        .vote-btn { background: #4caf50; color: white; border: none; padding: 8px 15px; cursor: pointer; border-radius: 4px; margin-top: 10px; width: 100%; font-weight: bold; }
        .vote-btn:hover { background: #45a049; }
        .card.selected .vote-btn { background: #2e7d32; content: "Selected"; }

        /* Floating Action Button */
        #fab { position: fixed; bottom: 20px; right: 20px; background: #61dafb; color: #000; border: none; padding: 15px 25px; border-radius: 30px; font-size: 1.1em; font-weight: bold; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); z-index: 900; }
        #fab:hover { background: #4fa8c7; }

        /* Modal for full images */
        #modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; align-items: center; justify-content: center; }
        #modal img { max-width: 95%; max-height: 95%; object-fit: contain; }
        #modal-close { position: absolute; top: 20px; right: 30px; color: white; font-size: 30px; cursor: pointer; font-weight: bold; }
        #modal-caption { position: absolute; bottom: 20px; color: white; font-size: 20px; background: rgba(0,0,0,0.5); padding: 5px 15px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Clarity Model Benchmark Viewer</h1>
    <p>Compare the 1:1 center crops across different models and combinations. Click any image to view the full resolution version. <b>Click 'Vote for this' to select the best output for each texture.</b></p>

    <button id="fab" onclick="exportVotes()">💾 Save Votes (clarity_votes.json)</button>

    <div class="tabs" id="view-tabs">
        <button class="tab-btn active" onclick="setView('crops')">View 1:1 Crops</button>
        <button class="tab-btn" onclick="setView('full')">View Full Images</button>
    </div>
"""
    ]

    for category in categories:
        cat_path = os.path.join(results_dir, category)
        specimens = sorted(
            [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
        )
        if not specimens:
            continue

        html.append(f'<div class="category-section" id="{category}">')
        html.append(f"<h2>{category}</h2>")

        for spec in specimens:
            spec_path = os.path.join(cat_path, spec)
            meta_path = os.path.join(spec_path, "metadata.json")
            meta_info = ""
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    meta_info = f"Format: {meta.get('format')} | Native Resolution: {meta.get('resolution')} | Game Path: {meta.get('path')}"

            html.append('<div class="specimen">')
            html.append(f'<div class="specimen-title">{spec}</div>')
            html.append(f'<div class="meta">{meta_info}</div>')
            html.append('<div class="grid">')

            # Find all crop pngs
            crops = glob.glob(os.path.join(spec_path, "*_crop.png"))
            crops.sort()

            # If no crops exist (maybe too small to crop), fallback to full
            if not crops:
                crops = glob.glob(os.path.join(spec_path, "*.png"))
                crops.sort()

            for crop in crops:
                basename = os.path.basename(crop)

                # Filter out irrelevant combinations to reduce clutter
                if "BC1" in category:
                    # For BC1, keep Vanilla, UpscalerV4 (to check double-smoothing), and the two smoothed variants
                    if not any(
                        x in basename
                        for x in [
                            "1_vanilla",
                            "3_UpscalerV4",
                            "5_BC1smooth2_RPLKSRd",
                            "6_BC1smooth2_UpscalerV4",
                        ]
                    ):
                        continue
                # For BC7, keep Vanilla, RPLKSRd, and UpscalerV4
                elif not any(x in basename for x in ["1_vanilla", "2_RPLKSRd_V3", "3_UpscalerV4"]):
                    continue

                full_name = basename.replace("_crop.png", ".png")

                title = (
                    basename.replace("_crop.png", "")
                    .replace(".png", "")
                    .replace("1_", "")
                    .replace("2_", "")
                    .replace("3_", "")
                    .replace("4_", "")
                    .replace("5_", "")
                    .replace("6_", "")
                    .replace("_", " ")
                )

                rel_crop = f"{category}/{spec}/{basename}"
                rel_full = f"{category}/{spec}/{full_name}"

                html.append(f'''
                <div class="card" id="card-{spec}-{title}">
                    <div class="card-title">{title}</div>
                    <img src="{rel_crop}" data-full="{rel_full}" data-crop="{rel_crop}" alt="{title}" class="compare-img" onclick="openModal(this)">
                    <button class="vote-btn" onclick="vote('{category}', '{spec}', '{title}', this)">Vote for this</button>
                </div>
                ''')

            html.append("</div></div>")
        html.append("</div>")

    html.append("""
    <div id="modal" onclick="closeModal()">
        <span id="modal-close">&times;</span>
        <img id="modal-img" src="">
        <div id="modal-caption"></div>
    </div>

    <script>
        let currentView = 'crops';
        let votes = {};

        function setView(view) {
            currentView = view;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            const imgs = document.querySelectorAll('.compare-img');
            imgs.forEach(img => {
                if (view === 'crops') {
                    img.src = img.getAttribute('data-crop');
                } else {
                    img.src = img.getAttribute('data-full');
                }
            });
        }

        function vote(category, spec, title, btnElement) {
            if (!votes[category]) {
                votes[category] = {};
            }
            votes[category][spec] = title;

            // Remove selected class from all cards in this grid
            const grid = btnElement.closest('.grid');
            grid.querySelectorAll('.card').forEach(card => {
                card.classList.remove('selected');
                card.querySelector('.vote-btn').innerText = 'Vote for this';
            });

            // Add selected class to chosen card
            const card = btnElement.closest('.card');
            card.classList.add('selected');
            btnElement.innerText = '★ Selected';

            console.log("Voted:", votes);
        }

        function exportVotes() {
            if (Object.keys(votes).length === 0) {
                alert("You haven't cast any votes yet!");
                return;
            }

            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(votes, null, 2));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "clarity_votes.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }

        function openModal(imgElement) {
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const caption = document.getElementById('modal-caption');

            modal.style.display = 'flex';
            modalImg.src = imgElement.getAttribute('data-full');
            caption.innerText = imgElement.alt;
        }

        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }

        document.addEventListener('keydown', function(event) {
            if (event.key === "Escape") {
                closeModal();
            }
        });
    </script>
</body>
</html>
    """)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Viewer generated at: {output_html}")


if __name__ == "__main__":
    generate_viewer(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS)
