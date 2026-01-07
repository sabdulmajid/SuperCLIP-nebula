import os
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import webdataset as wds
import wandb
from tqdm import tqdm
from torchvision import transforms

# --- Vanilla CLIP Model (No Super Head) ---
class VanillaCLIP(nn.Module):
    def __init__(self, base_model='ViT-B-32'):
        super().__init__()
        self.clip = open_clip.create_model(base_model, pretrained='openai')
        # We still bind these for consistency with our eval scripts
        self.visual = self.clip.visual
        self.text = self.clip.transformer
        self.token_embedding = self.clip.token_embedding
        self.ln_final = self.clip.ln_final
        self.logit_scale = self.clip.logit_scale
        # NOTE: No super_head here!

    def forward(self, image, text):
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text)
        return image_features, text_features, self.logit_scale.exp()

# --- Data (Same Augmented Loader) ---
def get_wds_loader(shards_url, batch_size, tokenizer, is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
        ])

    def preprocess(sample):
        image_bytes, text_data = sample
        text = text_data.get('caption') if isinstance(text_data, dict) else text_data
        try:
            return transform(image_bytes), tokenizer(text).squeeze(0)
        except:
            return None

    dataset = (
        wds.WebDataset(shards_url, resampled=True)
        .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "txt;json")
        .map(preprocess)
        .batched(batch_size, partial=False)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=8) 

# --- Main ---
def main():
    BATCH_SIZE = 384    
    EPOCHS = 20         
    LR = 5e-5           
    
    device = "cuda"
    os.makedirs("logs/checkpoints_baseline", exist_ok=True) # Separate folder
    wandb.init(project="superclip-nebula", name="vanilla-baseline", mode="offline")

    model = VanillaCLIP().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    dataloader = get_wds_loader("data/shards/train.tar", BATCH_SIZE, tokenizer, is_train=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * 100)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Training Vanilla Baseline (No SuperHead)...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(range(50), desc=f"Epoch {epoch+1}")
        data_iter = iter(dataloader)
        
        for _ in pbar:
            try:
                images, texts = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, texts = next(data_iter)
                
            images, texts = images.to(device), texts.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                img_emb, text_emb, logit_scale = model(images, texts)
                if logit_scale.ndim > 0: logit_scale = logit_scale[0]
                
                # --- STANDARD CLIP LOSS ONLY ---
                img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
                logits_per_image = logit_scale * img_emb @ text_emb.t()
                labels = torch.arange(img_emb.shape[0], device=img_emb.device)
                
                loss = (nn.functional.cross_entropy(logits_per_image, labels) + 
                        nn.functional.cross_entropy(logits_per_image.t(), labels)) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"logs/checkpoints_baseline/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()