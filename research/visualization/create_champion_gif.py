import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import io

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = SuperCLIP(base_model='ViT-B-32', vocab_size=49408)
    
    checkpoint_path = "../../logs/checkpoints/epoch_20.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "../logs/checkpoints/epoch_20.pt" 
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

    val_dir = "../../data/val2017"
    if not os.path.exists(val_dir): val_dir = "../data/val2017"
    
    target_file = "000000000139.jpg"
    img_path = os.path.join(val_dir, target_file)
    print(f"Processing {img_path}...")
    
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    original_image = Image.open(img_path).convert("RGB")
    original_image = original_image.resize((224, 224), Image.Resampling.BICUBIC)
    
    transform = get_transform()
    img_tensor = transform(original_image).unsqueeze(0).to(device)

    activations = {}
    
    for i, block in enumerate(model.visual.transformer.resblocks):
        if not hasattr(block.attn, 'original_forward'):
            block.attn.original_forward = block.attn.forward
            
        def make_new_forward(original_fwd, layer_id):
            def new_fwd(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
                out, weights = original_fwd(query, key, value, key_padding_mask, True, attn_mask, False, is_causal)
                activations[layer_id] = weights.detach().cpu()
                return out, weights
            return new_fwd
        
        block.attn.forward = make_new_forward(block.attn.original_forward, i)
    
    with torch.no_grad():
        model.visual(img_tensor)
        
    print("Generating frames...")
    frames = []
    
    os.makedirs("../plots", exist_ok=True)
    
    head_10_idx = 10 
    head_6_idx = 5
    
    num_layers = 12
    
    for layer_idx in range(num_layers):
        attn_map = activations[layer_idx]
        
        cls_attn_10 = attn_map[0, head_10_idx, 0, 1:]
        grid_10 = cls_attn_10.reshape(7, 7).numpy()
        heatmap_10 = (grid_10 - grid_10.min()) / (grid_10.max() - grid_10.min() + 1e-8)
        heatmap_10_res = cv2.resize(heatmap_10, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        cls_attn_6 = attn_map[0, head_6_idx, 0, 1:] 
        grid_6 = cls_attn_6.reshape(7, 7).numpy()
        heatmap_6 = (grid_6 - grid_6.min()) / (grid_6.max() - grid_6.min() + 1e-8)
        heatmap_6_res = cv2.resize(heatmap_6, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original_image)
        axes[0].imshow(heatmap_10_res, cmap='jet', alpha=0.5)
        axes[0].axis('off')
        
        axes[1].imshow(original_image)
        axes[1].imshow(heatmap_6_res, cmap='jet', alpha=0.5)
        axes[1].axis('off')
        
        role_text_left = ""
        role_text_right = ""
        color_left = "black"
        color_right = "black"
        bg_left = "white"
        bg_right = "white"
        
        if layer_idx == 6:
            role_text_left = "* DISCOVERY? *"
            color_left = "yellow"
            role_text_right = "(Analyzing...)"
        elif layer_idx == 11:
            role_text_left = "(Faded)"
            role_text_right = "* FOUND *"
            color_right = "lime"
            bg_right = "black"
        else:
            role_text_left = "Processing..."
            role_text_right = "Processing..."
            
        axes[0].set_title(f"Layer {layer_idx+1} Head {head_10_idx}\n(Old Hypothesis)\n{role_text_left}", 
                          fontsize=12, fontweight='bold', color=color_left, backgroundcolor=bg_left)
        axes[1].set_title(f"Layer {layer_idx+1} Head {head_6_idx}\n(New Champion)\n{role_text_right}", 
                          fontsize=12, fontweight='bold', color=color_right, backgroundcolor=bg_right)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        frame = Image.open(buf)
        frames.append(frame)
        print(f"  Processed Layer {layer_idx+1}")

    gif_path = "../plots/champion_vs_challenger.gif"
    durations = [300] * 12
    durations[6] = 1200
    durations[11] = 2000
    
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=durations, loop=0)
    print(f"\nDone! GIF saved to {gif_path}")

if __name__ == "__main__":
    main()
