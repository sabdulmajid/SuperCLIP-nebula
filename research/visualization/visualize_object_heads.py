import sys
import os
import torch
import torch.nn as nn
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = SuperCLIP(base_model='ViT-B-32', vocab_size=49408)
    model_path = "../../logs/checkpoints/epoch_20.pt"
    if not os.path.exists(model_path): model_path = "../logs/checkpoints/epoch_20.pt"
    
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    image_dir = "../../data/val2017"
    if not os.path.exists(image_dir): image_dir = "../data/val2017"
    target_file = "000000000139.jpg"
    
    path = os.path.join(image_dir, target_file)
    original_image = Image.open(path).convert("RGB")
    original_image = original_image.resize((224, 224), Image.Resampling.BICUBIC)
    
    transform = get_transform()
    img_tensor = transform(original_image).unsqueeze(0).to(device)

    print("Monkey-patching Attention forward to capture weights...")
    
    activations = {}

    first_block = model.visual.transformer.resblocks[0]
    attn_module = first_block.attn
    AttentionClass = attn_module.__class__
    
    original_forward = AttentionClass.forward
    
    def new_forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        output, weights = original_forward(
            self,
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,
            is_causal=is_causal
        )
        
        activations[self] = weights.detach().cpu()
        return output, weights

    AttentionClass.forward = new_forward
    
    with torch.no_grad():
        model.visual(img_tensor)
    
    AttentionClass.forward = original_forward
    
    print(f"Captured attention maps for {len(activations)} layers.")

    layer_map = {block.attn: i for i, block in enumerate(model.visual.transformer.resblocks)}
    
    target_layers = [5, 6, 7, 8]
    
    out_dir = "../plots/heads"
    os.makedirs(out_dir, exist_ok=True)
    
    for layer_idx in target_layers:
        block = model.visual.transformer.resblocks[layer_idx]
        attn = activations[block.attn] 
        
        cls_attn = attn[0, :, 0, 1:] 
        
        grid_size = int(np.sqrt(49))
        cls_attn_grid = cls_attn.reshape(12, grid_size, grid_size).cpu().numpy()
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle(f"Layer {layer_idx+1} [CLS] Attention", fontsize=16)
        
        for h in range(12):
            ax = axes[h // 4, h % 4]
            heatmap = cls_attn_grid[h]
            
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            img_np = np.array(original_image) / 255.0
            
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            
            overlay = 0.6 * img_np + 0.4 * heatmap_colored
            
            ax.imshow(overlay)
            ax.axis('off')
            ax.set_title(f"Head {h}")
            
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"layer_{layer_idx+1}_heads.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
