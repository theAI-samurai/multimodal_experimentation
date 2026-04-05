import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SimpleTextEncoder(nn.Module):
    """Your original text encoder"""
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        if mask is not None:
            # Convert to bool for Transformer (True = attend)
            mask = mask.bool()
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)  # mean pooling
        return self.proj(x)
    

class ModernBERTEncoder(nn.Module):
    """
    ModernBERT (answerdotai/ModernBERT-base or large)
    - Modern architecture with RoPE, GeGLU, better training recipe
    - Excellent general text understanding + strong embeddings
    """
    def __init__(self, model_name="answerdotai/ModernBERT-base", embed_dim=256, freeze_backbone=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(hidden_size, embed_dim)  # project to your embed_dim

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"Loaded ModernBERT: {model_name} | Hidden size: {hidden_size} → Projected to {embed_dim}")

    def forward(self, texts):  # texts = list of strings
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Use [CLS] token or mean pooling

        embeddings = outputs.last_hidden_state.mean(dim=1)   # or outputs.pooler_output
        return self.proj(embeddings)
    
class SigLIPTextEncoder(nn.Module):
    """
    SigLIP-style text encoder (uses the text tower from SigLIP models)
    - Trained with sigmoid loss on massive image-text data
    - Excellent for contrastive multimodal alignment (better than classic CLIP in many cases)
    """
    def __init__(self, model_name="google/siglip-base-patch16-224", embed_dim=256, freeze_backbone=False):
        super().__init__()
        # SigLIP models have both vision and text towers. We only load the text part.
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.model.config.text_config.hidden_size
        self.proj = nn.Linear(hidden_size, embed_dim)
        
        if freeze_backbone:
            for param in self.model.text_model.parameters():   # only freeze text tower
                param.requires_grad = False
        
        print(f"Loaded SigLIP Text Encoder: {model_name} | Hidden size: {hidden_size} → Projected to {embed_dim}")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(next(self.parameters()).device)

        # SigLIP text model is accessed via .text_model
        outputs = self.model.text_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)   # mean pooling
        
        return self.proj(embeddings)
    

# -----------------------------------------------
# # How to use ModernBERTEncoder

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# text_encoder = ModernBERTEncoder(model_name="answerdotai/ModernBERT-base", embed_dim=256).to(device)
# text_encoder = SigLIPTextEncoder(model_name="google/siglip-base-patch16-224", embed_dim=256, freeze_backbone=True).to(device)   # freeze for stability