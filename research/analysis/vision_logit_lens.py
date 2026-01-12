import sys
import os
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
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

def filename_to_id(fn):
    return int(os.path.splitext(fn)[0])

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
    ann_path = "../../data/annotations/captions_val2017.json"
    
    if not os.path.exists(image_dir): image_dir = "../data/val2017"
    if not os.path.exists(ann_path): ann_path = "../data/annotations/captions_val2017.json"

    image_files = sorted(os.listdir(image_dir))[:3]
    transform = get_transform()

    images = []
    print(f"\nProcessing {len(image_files)} images:")
    for img_file in image_files:
        path = os.path.join(image_dir, img_file)
        img = Image.open(path).convert("RGB")
        print(f" - {img_file}")
        images.append(transform(img).unsqueeze(0))
    
    batch = torch.cat(images).to(device)

    with open(ann_path) as f:
        coco_val = json.load(f)
    
    id_to_captions = {}
    for ann in coco_val['annotations']:
        img_id = ann['image_id']
        id_to_captions.setdefault(img_id, []).append(ann['caption'])

    intermediate_results = {}
    
    def get_activation(name):
        def hook(model, input, output):
            intermediate_results[name] = output.detach()
        return hook

    hooks = []
    for i, block in enumerate(model.visual.transformer.resblocks):
        hooks.append(block.register_forward_hook(get_activation(f"block_{i}")))

    with torch.no_grad():
        _ = model.clip.encode_image(batch)

    for h in hooks:
        h.remove()

    proj = model.visual.proj.detach()
    ln_post = model.visual.ln_post

    print("\n--- VISION LOGIT LENS RESULTS ---")
    
    # Special tokens to mask
    sot_token = tokenizer.encoder['<start_of_text>']
    eot_token = tokenizer.encoder['<end_of_text>']
    
    def decode_logits(logits, topk=10):
        logits[sot_token] = -float('inf')
        logits[eot_token] = -float('inf')
        
        probs = logits.sigmoid() 
        values, indices = probs.topk(topk)
        results = []
        for v, i in zip(values, indices):
            token = tokenizer.decode([i.item()])
            results.append(f"{token.strip()} ({v.item():.2f})")
        return ", ".join(results)

    layers_to_check = [1, 3, 5, 7, 9, 11] 
    
    for img_idx, img_file in enumerate(image_files):
        print(f"\nImage: {img_file}")
        img_id = filename_to_id(img_file)
        captions = id_to_captions.get(img_id, ["No caption found"])
        print(f"Ground Truth: {captions[0]}") 
        
        for layer_idx in layers_to_check:
            feat = intermediate_results[f"block_{layer_idx}"]
            img_feat = feat[img_idx:img_idx+1] 
            
            cls_token = img_feat[:, 0, :] 
            normalized = ln_post(cls_token) # Layer Norm
            projected = normalized @ proj   # Project to embedding space
            logits = model.super_head(projected) # Decode vocabulary
            
            print(f"Layer {layer_idx+1}: {decode_logits(logits[0])}")

if __name__ == "__main__":
    main()
