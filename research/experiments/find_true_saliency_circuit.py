import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

def load_coco_ground_truth(data_dir, num_images=100):
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    
    if not os.path.exists(ann_file):
        print(f"Annotations not found at {ann_file}")
        return []

    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    img_map = {img['id']: img for img in coco['images']}
    ann_map = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann['bbox'])
        
    results = []
    print(f"Selecting {num_images} images with objects...")
    
    count = 0
    for img_id in img_map:
        if img_id in ann_map and len(ann_map[img_id]) > 0:
            file_name = img_map[img_id]['file_name']
            path = os.path.join(img_dir, file_name)
            if os.path.exists(path):
                width = img_map[img_id]['width']
                height = img_map[img_id]['height']
                
                norm_boxes = []
                for (x, y, w, h) in ann_map[img_id]:
                    x1 = x / width
                    x2 = (x + w) / width
                    y1 = y / height
                    y2 = (y + h) / height
                    norm_boxes.append([x1, y1, x2, y2])
                    
                results.append({
                    'path': path,
                    'boxes': norm_boxes,
                    'id': img_id
                })
                count += 1
                if count >= num_images:
                    break
    return results

def compute_attention_mass_in_box(attn_map_7x7, boxes):
    total_mass = np.sum(attn_map_7x7)
    if total_mass == 0: return 0.0
    
    in_box_mass = 0.0
    h, w = attn_map_7x7.shape
    for r in range(h):
        for c in range(w):
            y_center = (r + 0.5) / h
            x_center = (c + 0.5) / w
            
            val = attn_map_7x7[r, c]
            
            is_inside = False
            for box in boxes:
                bx1, by1, bx2, by2 = box
                if bx1 <= x_center <= bx2 and by1 <= y_center <= by2:
                    is_inside = True
                    break
            
            if is_inside:
                in_box_mass += val
                
    return in_box_mass / total_mass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading SuperCLIP model...")
    model = SuperCLIP(base_model='ViT-B-32', vocab_size=49408)
    
    ckpt_path = "../../logs/checkpoints/epoch_20.pt"
    if not os.path.exists(ckpt_path): ckpt_path = "../logs/checkpoints/epoch_20.pt"
        
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    data_dir = "../../data"
    if not os.path.exists(os.path.join(data_dir, "annotations")): data_dir = "../data"
    
    dataset = load_coco_ground_truth(data_dir, num_images=100)
    if not dataset:
        print("No dataset loaded.")
        return

    print(f"Scanning 144 Heads across {len(dataset)} images...")
    
    head_scores = np.zeros((12, 12))
    valid_samples = 0
    activations = {}

    for i, block in enumerate(model.visual.transformer.resblocks):
        if not hasattr(block.attn, 'original_forward'):
            block.attn.original_forward = block.attn.forward
            
        def make_new_forward(original_fwd, layer_id):
            def new_fwd(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
                out, weights = original_fwd(query, key, value, key_padding_mask, True, attn_mask, False, is_causal)
                activations[layer_id] = weights.detach() 
                return out, weights
            return new_fwd
        
        block.attn.forward = make_new_forward(block.attn.original_forward, i)

    transform = get_transform()
    
    for item in tqdm(dataset, desc="Processing Images"):
        try:
            img = Image.open(item['path']).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            activations = {} 
            
            with torch.no_grad():
                model.visual(tensor)
                
            for L in range(12):
                if L not in activations: continue
                attn_map = activations[L][0] 
                
                for H in range(12):
                    cls_attn = attn_map[H, 0, 1:] 
                    grid = cls_attn.view(7, 7).cpu().numpy()
                    
                    ratio = compute_attention_mass_in_box(grid, item['boxes'])
                    head_scores[L, H] += ratio

            valid_samples += 1
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            continue

    if valid_samples == 0:
        print("No valid samples found!")
        return

    head_scores = head_scores / valid_samples
    
    os.makedirs("../plots", exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(head_scores, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"SuperCLIP Saliency Hit Rate (N={valid_samples})")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.savefig("../plots/true_saliency_circuit.png")
    print(f"\nSaved heatmap to ../plots/true_saliency_circuit.png")
    
    flat_indices = np.argsort(head_scores.ravel())[::-1]
    print("\n" + "="*40)
    print("TRUE SALIENCY CIRCUIT (Top 5)")
    print("="*40)
    for idx in flat_indices[:5]:
        L, H = np.unravel_index(idx, head_scores.shape)
        print(f"Layer {L+1}, Head {H+1}: {head_scores[L, H]*100:.1f}% In-Box Attention")
    print("="*40)

if __name__ == "__main__":
    main()
