import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from data_loader import CustomImageCaptionDataset


# ------------------- Simple Text Tokenizer (very basic) -------------------
class SimpleTokenizer:
    def __init__(self, max_length=32):
        self.max_length = max_length
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = 2
    
    def fit_on_texts(self, texts):
        idx = 2
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def encode(self, text):
        tokens = [self.vocab.get(w, 1) for w in text.lower().split()]
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)

# ------------------- Image Encoder (Simple CNN) -------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, embed_dim)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)

# ------------------- Text Encoder (Simple Transformer) -------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling
        return self.proj(x)

# ------------------- Contrastive Loss (InfoNCE / CLIP Loss) -------------------
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_features, text_features):
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Cosine similarity
        logits = (image_features @ text_features.T) / self.temperature
        
        # Symmetric loss
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = self.criterion(logits, labels)
        loss_t = self.criterion(logits.T, labels)
        return (loss_i + loss_t) / 2.0

# ------------------- Main Training Script -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths - UPDATE THESE
    image_dir = "mycustomdata/images/val2017"  # ← change to your actual image directory
    annotation_file = "mycustomdata/annotations/coco_val_captions.txt"   # ← change to your actual txt file name
    
    # Dataset & DataLoader
    dataset = CustomImageCaptionDataset(image_dir, annotation_file) #, max_samples=5000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    
    # Build simple tokenizer from all captions
    all_captions = []
    for captions_list in dataset.captions_dict.values():
        all_captions.extend(captions_list)
    tokenizer = SimpleTokenizer(max_length=32)
    tokenizer.fit_on_texts(all_captions)
    
    # Models
    image_encoder = ImageEncoder(embed_dim=256).to(device)
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size, embed_dim=256).to(device)
    loss_fn = CLIPLoss(temperature=0.07).to(device)
    
    optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)
    
    # Training loop
    # num_epochs controls how many full passes through the dataset we make.
    # More epochs = more learning, but risk of overfitting on small datasets.
    num_epochs = 5   # Start small, increase later
    for epoch in range(num_epochs):
        # Put both encoders in training mode.
        # This enables dropout and batch-norm updates (if any).
        image_encoder.train()
        text_encoder.train()
        total_loss = 0.0

        # Each iteration gives us one batch of images + matching captions.
        # tqdm wraps the dataloader to display a live progress bar.
        for images, captions, _ in tqdm(dataloader):
            images = images.to(device)

            # Convert each caption string into a fixed-length token tensor,
            # then stack into a 2-D tensor of shape [batch_size, max_length].
            text_tokens = torch.stack([tokenizer.encode(cap) for cap in captions]).to(device)
            
            # Forward
            image_features = image_encoder(images)
            text_features = text_encoder(text_tokens)
            
            loss = loss_fn(image_features, text_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    print("✅ Training finished! Tiny CLIP model trained on your dataset.")
    
    # Save models
    torch.save(image_encoder.state_dict(), "tiny_clip_image_encoder.pth")
    torch.save(text_encoder.state_dict(), "tiny_clip_text_encoder.pth")
    print("Models saved!")