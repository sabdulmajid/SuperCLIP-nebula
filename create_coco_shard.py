import webdataset as wds
import json
import os
import random
from tqdm import tqdm

# Config
IMG_DIR = "data/val2017"
ANNOTATION_FILE = "data/annotations/captions_val2017.json"
TRAIN_TAR = "data/shards/train.tar"
VAL_TAR = "data/shards/val.tar"

# Load Data
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)

img_map = {img['id']: img['file_name'] for img in data['images']}
caption_map = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in caption_map:
        caption_map[img_id] = ann['caption']

# Split (80% Train, 20% Val) -- I used 5000 images with a 4000/1000 split, you can do more with more compute!
all_ids = list(img_map.keys())
random.seed(42)
random.shuffle(all_ids)
split_idx = int(len(all_ids) * 0.8)
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

print(f"Split: {len(train_ids)} Train, {len(val_ids)} Val")

# Shard Writer (Helper)
def write_shard(filename, ids):
    with wds.TarWriter(filename) as sink:
        for img_id in tqdm(ids, desc=f"Writing {filename}"):
            fname = img_map[img_id]
            caption = caption_map.get(img_id)
            if not caption or not os.path.exists(os.path.join(IMG_DIR, fname)):
                continue
            
            with open(os.path.join(IMG_DIR, fname), "rb") as stream:
                sample = {
                    "__key__": str(img_id),
                    "jpg": stream.read(),
                    "txt": caption
                }
                sink.write(sample)

write_shard(TRAIN_TAR, train_ids)
write_shard(VAL_TAR, val_ids)
print("ETL Complete.")