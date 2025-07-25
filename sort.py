import os
import shutil
import pandas as pd

# Paths
csv_path = "cassava-leaf-disease-classification/train.csv"
images_path = "cassava-leaf-disease-classification/train_images"
output_dir = "farmguard_dataset/cassava"

# Create folders
os.makedirs(f"{output_dir}/healthy", exist_ok=True)
os.makedirs(f"{output_dir}/mosaic_virus", exist_ok=True)
os.makedirs(f"{output_dir}/brown_streak", exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Label to folder mapping
label_map = {
    1: "brown_streak",
    3: "mosaic_virus",
    4: "healthy"
}

# Filter and copy
count = 0
for _, row in df.iterrows():
    label = row['label']
    if label in label_map:
        src = os.path.join(images_path, row['image_id'])
        dst = os.path.join(output_dir, label_map[label], row['image_id'])
        shutil.copy(src, dst)
        count += 1

print(f"✅ {count} images sorted into folders.")
