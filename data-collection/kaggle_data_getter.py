import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dest = os.path.join(project_root, "data-collection", "raw_data")

os.makedirs(dest, exist_ok=True)

for root, _, files in os.walk(path):
    for fname in files:
        src = os.path.join(root, fname)
        dst = os.path.join(project_root, dest, fname)
        shutil.copy2(src, dst)

print("Kaggle dataset cached at:", path)
print("Raw data copied to:", raw_data_dir)