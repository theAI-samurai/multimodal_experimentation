import os
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from data_loader import CustomImageCaptionDataset
from encoder_image import SimpleCNN, ResNetEncoder
from encoder_text import SimpleTextEncoder, BERTRoBERTaEncoder, ModernBERTEncoder, SigLIPTextEncoder
from tokenizer import SimpleTokenizer
from clip import CLIPLoss

load_dotenv()

TEXT_ENCODER     = os.getenv("TEXT_ENCODER", "roberta").lower()
TEXT_MODEL       = os.getenv("TEXT_ENCODER_MODEL", "roberta-base")
EMBED_DIM        = int(os.getenv("EMBED_DIM", "256"))
FREEZE_BACKBONE  = os.getenv("FREEZE_BACKBONE", "false").lower() == "true"
POOLING_STRATEGY = os.getenv("POOLING_STRATEGY", "mean")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Text encoder: {TEXT_ENCODER}" + (f" ({TEXT_MODEL})" if TEXT_ENCODER != "simple" else ""))

    image_dir = "mycustomdata/images/val2017/"
    annotation_file = "mycustomdata/annotations/coco_val_captions.txt"

    dataset = CustomImageCaptionDataset(image_dir, annotation_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # === Build text encoder from .env config ===
    use_raw_text = True   # HuggingFace encoders accept list[str] directly
    tokenizer = None

    if TEXT_ENCODER == "simple":
        all_captions = [c for caps in dataset.captions_dict.values() for c in caps]
        tokenizer = SimpleTokenizer(max_length=32)
        tokenizer.fit_on_texts(all_captions)
        text_encoder = SimpleTextEncoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=EMBED_DIM,
            num_heads=8,
            num_layers=4,
        ).to(device)
        use_raw_text = False
    elif TEXT_ENCODER in ("roberta", "bert"):
        text_encoder = BERTRoBERTaEncoder(
            model_name=TEXT_MODEL,
            embed_dim=EMBED_DIM,
            pooling_strategy=POOLING_STRATEGY,
            freeze_backbone=FREEZE_BACKBONE,
        ).to(device)
    elif TEXT_ENCODER == "modernbert":
        print("Using ModernBERT text encoder with custom pooling and projection.")
        text_encoder = ModernBERTEncoder(
            model_name=TEXT_MODEL,
            embed_dim=EMBED_DIM,
            freeze_backbone=FREEZE_BACKBONE,
        ).to(device)
    elif TEXT_ENCODER == "siglip":
        print("Using SigLIP text encoder with custom pooling and projection.")
        text_encoder = SigLIPTextEncoder(
            model_name=TEXT_MODEL,
            embed_dim=EMBED_DIM,
            freeze_backbone=FREEZE_BACKBONE,
        ).to(device)
    else:
        raise ValueError(f"Unknown TEXT_ENCODER='{TEXT_ENCODER}'. Choose: simple | roberta | bert | modernbert | siglip")

    image_encoder = ResNetEncoder(embed_dim=EMBED_DIM).to(device)

    loss_fn = CLIPLoss(temperature=0.07, learnable_temp=True).to(device)
    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters()) + list(text_encoder.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )
    scaler = GradScaler()

    num_epochs = 10
    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, captions, _ in tqdm(dataloader):
            images = images.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                image_features = image_encoder(images)
                if use_raw_text:
                    text_features = text_encoder(list(captions))
                else:
                    text_tokens = tokenizer.encode_batch(captions).to(device)
                    attn_mask = tokenizer.get_attention_mask(text_tokens).to(device)
                    text_features = text_encoder(text_tokens, mask=attn_mask)
                loss = loss_fn(image_features, text_features)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(image_encoder.parameters()) + list(text_encoder.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Quick accuracy
            with torch.no_grad():
                sim = (image_features @ text_features.T)
                preds = sim.argmax(dim=1)
                correct += (preds == torch.arange(len(preds), device=device)).sum().item()
                total += len(preds)

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - Batch Acc: {acc:.4f}")

    # Save
    torch.save(image_encoder.state_dict(), "image_encoder.pth")
    torch.save(text_encoder.state_dict(), f"text_encoder_{TEXT_ENCODER}.pth")
    print("Training completed and models saved!")