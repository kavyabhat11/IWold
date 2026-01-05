import os
import subprocess
import shutil

INPUT_DIR = "data/mozart"
OUTPUT_DIR = "analysis/mozart"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith(".xml"):
        continue
    if "-analysis" in fname:
        continue

    # full path to input file
    in_path = os.path.join(INPUT_DIR, fname)
    
    # base name (e.g., "K310-1" from "K310-1.xml")
    base = os.path.splitext(fname)[0]
    print(base)

    print(f"Analyzing {fname}...")

    # Run your analyzer normally (DEFAULT OUTPUT)
    subprocess.run(["python", "analyse_score.py", "--score_path", in_path], check=True)

    # The analyzer should have created:  <base>-analysis.musicxml
    out_filename = f"{base}-analysis.musicxml"

    out_path = os.path.join(INPUT_DIR, out_filename)

    if not os.path.exists(out_path):
        print(f"  ERROR: Expected output {out_filename} not found.")
        continue

    # Move output file to analysis/mozart/
    dest_path = os.path.join(OUTPUT_DIR, out_filename)
    shutil.move(out_path, dest_path)

    print(f"  → Moved {out_filename} to {dest_path}")
