import sys
import os
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from torchvision import transforms
import json

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
    if not os.path.exists(checkpoint_path): checkpoint_path = "../logs/checkpoints/epoch_20.pt"
    
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image_dir = "../../data/val2017"
    if not os.path.exists(image_dir): image_dir = "../data/val2017"
    
    transform = get_transform()
    
    data = [
        {"file": "000000000139.jpg", "concepts": ["woman", "table"]},
        {"file": "000000000285.jpg", "concepts": ["bear", "grass"]},
        {"file": "000000000632.jpg", "concepts": ["bedroom", "bookcase"]}
    ]

    print("\n--- CONCEPT EMERGENCE ANALYSIS ---")

    def get_token_id(word):
        tokens = tokenizer(word)[0] 
        if len(tokens) == 3:
            return tokens[1].item()
        else:
            return tokens[1].item()

    intermediate_results = {}
    def get_activation(name):
        def hook(model, input, output):
            intermediate_results[name] = output.detach()
        return hook

    hooks = []
    for i, block in enumerate(model.visual.transformer.resblocks):
        hooks.append(block.register_forward_hook(get_activation(f"block_{i}")))

    images = []
    for item in data:
        path = os.path.join(image_dir, item["file"])
        try:
            img = Image.open(path).convert("RGB")
            images.append(transform(img).unsqueeze(0))
        except Exception as e:
            print(f"Could not load {path}: {e}")
            
    if not images:
        print("No images found.")
        return
        
    batch = torch.cat(images).to(device)

    with torch.no_grad():
        _ = model.clip.encode_image(batch)

    for h in hooks: h.remove()

    proj = model.visual.proj.detach()
    ln_post = model.visual.ln_post
    
    layers = range(12)

    for i, item in enumerate(data):
        print(f"\nImage: {item['file']}")
        print(f"Tracking concepts: {item['concepts']}")
        
        target_ids = [(c, get_token_id(c)) for c in item['concepts']]
        print(f"Token IDs: {target_ids}")

        for token_word, token_id in target_ids:
            print(f"  Concept: '{token_word}'")
            print(f"    Layer | Rank  | Prob")
            print(f"    ------+-------+-------")
            
            for layer_idx in layers:
                if f"block_{layer_idx}" not in intermediate_results: continue
                
                feat = intermediate_results[f"block_{layer_idx}"][i:i+1, 0, :] 
                normalized = ln_post(feat)
                projected = normalized @ proj
                logits = model.super_head(projected).squeeze(0)
                
                probs = logits.sigmoid()
                
                score = probs[token_id].item()
                rank = (probs > score).sum().item() + 1
                
                print(f"    {layer_idx+1:2d}    | {rank:5d} | {score:.4f}")

if __name__ == "__main__":
    main()
