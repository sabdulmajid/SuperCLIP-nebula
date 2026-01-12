import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import glob
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SuperCLIP(base_model='ViT-B-32', vocab_size=49408)
    
    checkpoint_path = "../../logs/checkpoints/epoch_20.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "../logs/checkpoints/epoch_20.pt"
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    val_dir = "../../data/val2017"
    if not os.path.exists(val_dir): val_dir = "../data/val2017"
    all_images = glob.glob(os.path.join(val_dir, "*.jpg"))
    if not all_images:
        print(f"No images found in {val_dir}")
        return
        
    selected_paths = random.sample(all_images, min(5, len(all_images)))
    
    TARGET_LAYER = 11 
    TARGET_HEAD = 5

    activations = {}
    block = model.visual.transformer.resblocks[TARGET_LAYER]
    
    if not hasattr(block.attn, 'original_forward'):
        block.attn.original_forward = block.attn.forward
        
    def new_fwd(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        out, weights = block.attn.original_forward(query, key, value, key_padding_mask, True, attn_mask, False, is_causal)
        activations["attn"] = weights.detach().cpu()
        return out, weights
    
    block.attn.forward = new_fwd
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"The Champion: Layer {TARGET_LAYER+1} Head {TARGET_HEAD+1} (81% Accuracy)", fontsize=16)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

    print("Generating Money Shot...")
    for i, img_path in enumerate(selected_paths):
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((224, 224))
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            model.visual(tensor)
            
        attn = activations["attn"][0, TARGET_HEAD, 0, 1:]
        
        grid = attn.reshape(7, 7).numpy()
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        heatmap = cv2.resize(grid, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        axes[i].imshow(img_resized)
        axes[i].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i].axis('off')
        axes[i].set_title(os.path.basename(img_path), fontsize=8)

    os.makedirs("../plots", exist_ok=True)
    save_path = "../plots/champion_head_visualization.png"
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
