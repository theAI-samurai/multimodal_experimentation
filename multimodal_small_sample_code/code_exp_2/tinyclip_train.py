import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from data_loader import CustomImageCaptionDataset
from encoder_image import SimpleCNN, ResNetEncoder
from encoder_text import SimpleTextEncoder
from tokenizer import SimpleTokenizer
from clip import CLIPLoss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir = "mycustomdata/images/val2017/"
    annotation_file = "mycustomdata/annotations/coco_val_captions.txt"

    dataset = CustomImageCaptionDataset(image_dir, annotation_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# Tokenizer
    all_captions = []
    for captions_list in dataset.captions_dict.values():
        all_captions.extend(captions_list)
    tokenizer = SimpleTokenizer(max_length=32)
    tokenizer.fit_on_texts(all_captions)

    # === Choose your encoders here ===
    image_encoder = ResNetEncoder(embed_dim=256).to(device)        # or SimpleCNN(embed_dim=256)
    text_encoder = SimpleTextEncoder(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=256, 
        num_heads=8, 
        num_layers=4
    ).to(device)

    loss_fn = CLIPLoss(temperature=0.07, learnable_temp=True).to(device)
    optimizer = torch.optim.AdamW(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()   # for mixed precision

    num_epochs = 10
    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, captions, _ in tqdm(dataloader):
            images = images.to(device)
            text_tokens = tokenizer.encode_batch(captions).to(device)
            attn_mask = tokenizer.get_attention_mask(text_tokens).to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                image_features = image_encoder(images)
                text_features = text_encoder(text_tokens, mask=attn_mask)
                loss = loss_fn(image_features, text_features)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(image_encoder.parameters()) + 
                                        list(text_encoder.parameters()), max_norm=1.0)
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
    torch.save(text_encoder.state_dict(), "text_encoder.pth")
    print("Training completed and models saved!")