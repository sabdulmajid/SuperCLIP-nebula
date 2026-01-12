import sys
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP
import open_clip

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
    if not os.path.exists(model_path):
        model_path = "../logs/checkpoints/epoch_20.pt"
        
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

    val_dir = "../../data/val2017" 
    if not os.path.exists(val_dir):
         val_dir = "../data/val2017"
         
    all_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    random.seed(42) 
    selected_images = random.sample(all_images, 5)
    
    print(f"Selected images: {selected_images}")
    
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
        
        if self not in activations:
            activations[self] = []
        activations[self].append(weights.detach().cpu())
        
        return output, weights

    AttentionClass.forward = new_forward
    
    transform = get_transform()
    
    processed_images = []
    
    print("Running inference...")
    with torch.no_grad():
        for img_file in selected_images:
            path = os.path.join(val_dir, img_file)
            img = Image.open(path).convert("RGB")
            processed_images.append(img.resize((224, 224)))
            
            tensor = transform(img).unsqueeze(0).to(device)
            model.visual(tensor)

    AttentionClass.forward = original_forward
    
    layer_map = {block.attn: i for i, block in enumerate(model.visual.transformer.resblocks)}
    
    out_dir = "../plots/heads_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    target_layers = [6, 7] 
    target_head = 10 
    
    for layer_idx in target_layers:
        block = model.visual.transformer.resblocks[layer_idx]
        attn_list = activations[block.attn] 
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(f"Layer {layer_idx+1} Head {target_head}: Object Head Validation", fontsize=20)
        
        for i, ax in enumerate(axes):
            attn = attn_list[i] 
            cls_attn = attn[0, target_head, 0, 1:] 
            
            grid_size = int(np.sqrt(49))
            heatmap = cls_attn.reshape(grid_size, grid_size).numpy()
            
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            img_np = np.array(processed_images[i]) / 255.0
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            
            overlay = 0.6 * img_np + 0.4 * heatmap_colored
            
            ax.imshow(overlay)
            ax.axis('off')
            ax.set_title(f"Image {i+1}\n{selected_images[i]}")
            
        save_path = os.path.join(out_dir, f"layer_{layer_idx+1}_head_{target_head}_validation.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved validation plot: {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
