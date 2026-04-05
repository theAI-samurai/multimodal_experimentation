"""
DataLoader.py

Builds a ``CustomImageCaptionDataset`` from a local COCO-style dataset and
wraps it in a PyTorch ``DataLoader`` for batched, shuffled iteration.

The dataset pairs images (JPEG/PNG) with their text captions read from a
``file_name|caption`` annotation file produced by ``coco_dataset_convert.py``.

Typical usage::

    python DataLoader.py

Configuration (edit the constants near the bottom of this file):
    image_dir       -- path to the folder that contains image files
    annotation_file -- path to the ``file_name|caption`` text file
    batch_size      -- number of samples per batch (default: 32)
    num_workers     -- parallel data-loading workers (default: 4)
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageCaptionDataset(Dataset):
    """
    PyTorch Dataset that pairs local images with their text captions.

    Each sample is a ``(image_tensor, caption, img_id)`` tuple where
    ``image_tensor`` is a transformed ``torch.Tensor``, ``caption`` is a
    randomly selected string from all available captions for that image, and
    ``img_id`` is the image filename (useful for debugging).
    """

    def __init__(self, image_dir: str, annotation_file: str, transform=None, max_samples=None):
        """
        Initialise the dataset.

        Args:
            image_dir (str): Path to the folder containing image files
                (e.g. ``mycustomdata/images/val2017``).
            annotation_file (str): Path to the ``file_name|caption`` text file
                (e.g. ``mycustomdata/annotations/coco_val_captions.txt``).
            transform (callable, optional): Torchvision transform to apply to
                each image.  Defaults to resize \u2192 tensor \u2192 ImageNet-normalise.
            max_samples (int, optional): If set, only the first *max_samples*
                images are used (handy for quick smoke-tests).
        """
        self.image_dir = image_dir
        self.transform = transform or self.default_transform()
        
        # Load annotations: image_id -> list of captions
        self.captions_dict = {}
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id, caption = line.split('|', 1)
                img_id = img_id.strip()
                caption = caption.strip()
                if img_id not in self.captions_dict:
                    self.captions_dict[img_id] = []
                self.captions_dict[img_id].append(caption)
        
        self.image_ids = list(self.captions_dict.keys())
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        print(f"Dataset loaded: {len(self.image_ids)} unique images, "
              f"~{len(self.image_ids)*5} total captions")
    
    def default_transform(self):
        """Return the default image transform pipeline (resize → tensor → normalize)."""
        return transforms.Compose([
            transforms.Resize((224, 224)),          # Good starting size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        """Return the number of unique images in the dataset."""
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Return the (image_tensor, caption, img_id) tuple for the given index.

        A random caption is selected when multiple captions exist for an image.
        Raises ``FileNotFoundError`` if the image file cannot be located.
        """
        img_id = self.image_ids[idx]

        
        # Load image - assume filename is exactly the image_id (e.g. 123456.jpg)
        # If your filenames have extensions or prefixes, adjust here
        # possible_extensions = ['.jpg', '.jpeg', '.png']
        img_path = None
        # for ext in possible_extensions:
        candidate = os.path.join(self.image_dir, f"{img_id}")
        if os.path.exists(candidate):
            img_path = candidate

        if img_path is None:
            # Fallback: search for any file starting with img_id
            for fname in os.listdir(self.image_dir):
                if fname.startswith(str(img_id)):
                    img_path = os.path.join(self.image_dir, fname)
                    break
        
        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for id: {img_id}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Randomly pick one caption (common practice)
        caption = random.choice(self.captions_dict[img_id])
        
        return image, caption, img_id   # return img_id for debugging if needed


# # ---------------------------------------------------------------------------
# # Dataset
# # ---------------------------------------------------------------------------
# # ``CustomImageCaptionDataset`` implements PyTorch's ``Dataset`` interface so
# # it can be indexed like a list.  Internally it:
# #   1. Reads the annotation file into a dict  { filename -> [caption, ...] }
# #   2. On each ``__getitem__`` call it:
# #        • resolves the image file path from ``image_dir``
# #        • opens and converts the image to RGB via Pillow
# #        • applies the transform pipeline (resize → tensor → normalise)
# #        • picks one caption at random (standard practise for image-caption tasks)
# #   The returned tuple is  (image_tensor [C,H,W], caption str, img_id str).
# # ---------------------------------------------------------------------------
# dataset = CustomImageCaptionDataset(
#     image_dir="mycustomdata/images/val2017",
#     annotation_file="mycustomdata/annotations/coco_val_captions.txt",  # ← change this
#     #max_samples=100  # for fast testing
# )

# # # test Dataset indexing and output shapes
# # img, cap, iid = dataset[0]
# # print("Image shape:", img.shape)
# # print("Caption:", cap)
# # print("Image ID:", iid)


# # ---------------------------------------------------------------------------
# # DataLoader
# # ---------------------------------------------------------------------------
# # ``DataLoader`` wraps the dataset and handles:
# #   • Batching      – collects ``batch_size`` samples into a single tensor
# #                     (images → [B, C, H, W]; captions/ids → lists of strings)
# #   • Shuffling     – randomises sample order each epoch to reduce overfitting
# #   • Multi-process loading – ``num_workers`` background workers pre-fetch
# #                     batches in parallel, keeping the GPU busy
# #   • Pinned memory – ``pin_memory=True`` locks CPU buffers so GPU transfers
# #                     via ``.to(device, non_blocking=True)`` are faster
# # ---------------------------------------------------------------------------
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,      # increase if you have good CPU
#     pin_memory=True     # faster GPU transfer
# )

# # batch = next(iter(dataloader))
# # images, captions, img_ids = batch
# # print("Batch images shape:", images.shape)   # [32, 3, 224, 224]
# # print("Number of captions in batch:", len(captions))
