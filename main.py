import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("nelgiriyewithana/global-weather-repository")
print("Path to dataset files:", path)

# Copy dataset to data/raw/
raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

for f in os.listdir(path):
    src = os.path.join(path, f)
    if os.path.isfile(src) and not f.endswith(".db"):
        dst = os.path.join(raw_dir, f)
        shutil.copy2(src, dst)
        print(f"Copied {f} -> {dst}")