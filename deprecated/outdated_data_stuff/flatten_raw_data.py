import os
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "raw_data"))

for root, dirs, files in os.walk(BASE_DIR):
    # skip the top-level raw_data itself
    if root == BASE_DIR:
        continue

    for fname in files:
        src = os.path.join(root, fname)
        dst = os.path.join(BASE_DIR, fname)

        # if a file with the same name already exists at top level, skip to be safe
        if os.path.exists(dst):
            print(f"Skipping {src} -> {dst} (already exists)")
            continue

        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

# after moving, remove now-empty subdirectories (a, b, c, ...)
for root, dirs, files in os.walk(BASE_DIR, topdown=False):
    if root == BASE_DIR:
        continue
    if not os.listdir(root):
        print(f"Removing empty directory {root}")
        os.rmdir(root)