import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import webdataset as wds
import wandb
from tqdm import tqdm
from torchvision import transforms

# Model
class SuperCLIP(nn.Module):
    def __init__(self, base_model='ViT-B-32', vocab_size=49408):
        super().__init__()
        self.clip = open_clip.create_model(base_model, pretrained='openai')
        self.visual = self.clip.visual
        self.text = self.clip.transformer
        self.token_embedding = self.clip.token_embedding
        self.ln_final = self.clip.ln_final
        self.logit_scale = self.clip.logit_scale
        
        visual_dim = self.visual.output_dim
        self.super_head = nn.Linear(visual_dim, vocab_size)

    def forward(self, image, text):
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text)
        token_logits = self.super_head(image_features)
        return image_features, text_features, self.logit_scale.exp(), token_logits

# Data Loading + Preprocessing
def get_wds_loader(shards_url, batch_size, tokenizer, is_train=True):
    # AUGMENTATION: This increases Recall by making the model robust
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Force it to look at details
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Robustness
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
        if isinstance(text_data, dict):
            text = text_data.get('caption') or str(text_data)
        else:
            text = text_data
        
        try:
            image = transform(image_bytes)
            text_tokens = tokenizer(text).squeeze(0)
            return image, text_tokens
        except:
            return None

    dataset = (
        wds.WebDataset(shards_url, resampled=True)
        .shuffle(5000) # Increased shuffle buffer for randomness
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "txt;json")
        .map(preprocess)
        .batched(batch_size, partial=False)
    )
    
    return wds.WebLoader(dataset, batch_size=None, num_workers=8) 

# Loss
def superclip_loss(img_emb, text_emb, logit_scale, token_logits, input_ids):
    img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
    logits_per_image = logit_scale * img_emb @ text_emb.t()
    labels = torch.arange(img_emb.shape[0], device=img_emb.device)
    loss_con = (nn.functional.cross_entropy(logits_per_image, labels) + 
                nn.functional.cross_entropy(logits_per_image.t(), labels)) / 2
    
    targets = torch.zeros(img_emb.shape[0], token_logits.shape[1], device=img_emb.device)
    targets.scatter_(1, input_ids, 1.0)
    loss_sup = nn.functional.binary_cross_entropy_with_logits(token_logits, targets)
    return loss_con + loss_sup, loss_con, loss_sup

# Main Training Loop
def main():
    # SETTINGS FOR MAX COMPUTE
    BATCH_SIZE = 384    # Tripled from 128 -> High GPU Util
    EPOCHS = 20         # Longer training for better convergence
    LR = 5e-5           # Lower LR since we are fine-tuning
    
    device = "cuda"
    os.makedirs("logs/checkpoints", exist_ok=True)
    wandb.init(project="superclip-nebula", mode="offline")

    model = SuperCLIP().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    dataloader = get_wds_loader("data/shards/train.tar", BATCH_SIZE, tokenizer, is_train=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.2)
    
    # SCHEDULER: Cosine Decay (standard for state-of-the-art results)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * 100)
    
    scaler = torch.cuda.amp.GradScaler()

    print(f"Training with Batch Size {BATCH_SIZE} on {torch.cuda.device_count()} GPUs")

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
                img_emb, text_emb, logit_scale, token_logits = model(images, texts)
                if logit_scale.ndim > 0: logit_scale = logit_scale[0]
                
                loss, l_con, l_sup = superclip_loss(img_emb, text_emb, logit_scale, token_logits, texts)
                if loss.dim() > 0: loss = loss.mean() # DataParallel fix

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Save epochs with multiples of 5 to save space
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"logs/checkpoints/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
