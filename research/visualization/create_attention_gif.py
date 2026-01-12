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
    
    block_0 = model.visual.transformer.resblocks[0]
    AttentionClass = block_0.attn.__class__
    original_forward = AttentionClass.forward
    
    def new_forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        output, weights = original_forward(
            self, query, key, value,
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
    
    print("Generating frames...")
    frames = []
    
    blocks = model.visual.transformer.resblocks
    
    # Updated plots path
    os.makedirs("../plots", exist_ok=True)
    
    head_idx = 10 
    
    for layer_idx, block in enumerate(blocks):
        attn_map = activations[block.attn] 
        cls_attn = attn_map[0, head_idx, 0, 1:] 
        
        grid_size = int(np.sqrt(49))
        heatmap = cls_attn.reshape(grid_size, grid_size).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        
        ax.imshow(original_image)
        ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
        
        role_text = "Processing..."
        color = "black"
        bg_color = "white"
        
        if layer_idx == 6: 
            role_text = "* DISCOVERY *"
            color = "black"
            bg_color = "white"
        elif layer_idx == 7: 
            role_text = "* LOCKED ON *"
            color = "red"
            bg_color = "white"
        elif layer_idx == 11: 
            role_text = "FOUND"
            color = "black"
            bg_color = "white"
        elif layer_idx < 6:
            role_text = "Attention Mechanism"
        
        title = ax.set_title(f"Layer {layer_idx+1} - Head {head_idx}\n{role_text}", fontsize=15, fontweight='bold', color=color)
        title.set_backgroundcolor(bg_color)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        frame = Image.open(buf)
        frames.append(frame)
        print(f"  Processed Layer {layer_idx+1}")

    gif_path = "../plots/attention_evolution.gif"
    durations = [400] * len(frames)
    durations[6] = 2000 
    durations[7] = 2000 
    durations[11] = 4000 
    
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=durations, loop=0)
    print(f"\nDone! GIF saved to {gif_path}")

if __name__ == "__main__":
    main()
