"""
coco_dataset_convert.py

Converts COCO-format annotation JSON into a flat text file of
``file_name|caption`` pairs, one per line.  The output can be consumed
directly by a PyTorch Dataset (see preprocess.py).

Usage:
    python coco_dataset_convert.py

Configuration (edit the constants at the top of the file):
    ann_file    -- path to the COCO-format captions JSON
    output_file -- destination for the produced text file
    max_images  -- optional integer cap on the number of images processed
                   (set to None to process all images)
"""

import json
import os

ann_file = 'mycustomdata/annotations/captions_val2017.json'   # Path to the JSON file
output_file = 'mycustomdata/annotations/coco_val_captions.txt'                 # Output text file name

max_images = None

# Load the annotations JSON
print(f"Loading annotations from: {ann_file}")
with open(ann_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create mapping: image_id -> list of captions
captions_dict = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    caption = ann['caption'].strip().replace('\n', ' ').replace('\r', ' ')

    file_name = ""
    for img in data["images"]:
        if img["id"] == img_id:
            file_name = img["file_name"]
            break

    if file_name not in captions_dict.keys():
        captions_dict[file_name] = []
    captions_dict[file_name].append(caption)

print(f"Found {len(captions_dict)} images with captions.")

# Optional limit
image_ids = list(captions_dict.keys())
if max_images is not None:
    image_ids = image_ids[:max_images]
    print(f"Limiting to first {max_images} images for testing.")

# Write to text file
print(f"Writing captions to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    for img_id in image_ids:
        for caption in captions_dict[img_id]:
            # Write: image_id|caption
            f.write(f"{img_id}|{caption}\n")

print(f"✅ Done! Created {output_file}")
print(f"Total lines (captions): {sum(len(captions_dict[img_id]) for img_id in image_ids)}")