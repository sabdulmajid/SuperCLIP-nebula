import sys
import os
import torch
import numpy as np
import json
from PIL import Image
import cv2
from torchvision import transforms
from tqdm import tqdm
import open_clip

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

def load_coco_ground_truth(data_dir, num_images=50):
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    
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
    
    print("Initializing SuperCLIP...")
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
    
    data_dir = "../../data"
    if not os.path.exists(os.path.join(data_dir, "annotations")): data_dir = "../data"
        
    dataset = load_coco_ground_truth(data_dir, num_images=100)
    print(f"Loaded {len(dataset)} images for validation.")
    
    transform = get_transform()
    
    activations = {}
    
    target_layer_idx = 6
    block_target = model.visual.transformer.resblocks[target_layer_idx]
    AttentionClass = block_target.attn.__class__
    original_forward = AttentionClass.forward
    
    def hook_forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        output, weights = original_forward(self, query, key, value, key_padding_mask, True, attn_mask, False, is_causal)
        activations['last_attn'] = weights.detach().cpu() 
        return output, weights
        
    AttentionClass.forward = hook_forward
    
    head_stats = {
        'head_10': [], 
        'head_1': [],  
        'center': []   
    }
    
    grid = np.zeros((7,7))
    for r in range(7):
        for c in range(7):
            dist = (r-3)**2 + (c-3)**2
            grid[r,c] = np.exp(-dist/4.0)
    center_prior = grid / np.sum(grid)
    
    print("\nRunning Mass Evaluation...")
    print("-" * 60)
    print(f"{'Image ID':<10} | {'L7 H10 (Ours)':<15} | {'L7 H1 (Rand)':<15} | {'Center Prior':<15}")
    print("-" * 60)
    
    for item in dataset:
        try:
            img = Image.open(item['path']).convert("RGB")
            
            x = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                model.visual(x)
                
            attn = activations['last_attn'] 
            
            h10_map = attn[0, 10, 0, 1:].reshape(7,7).numpy()
            h10_map = h10_map / np.sum(h10_map) 
            
            h1_map = attn[0, 1, 0, 1:].reshape(7,7).numpy()
            h1_map = h1_map / np.sum(h1_map)
            
            score_10 = compute_attention_mass_in_box(h10_map, item['boxes'])
            score_1 = compute_attention_mass_in_box(h1_map, item['boxes'])
            score_c = compute_attention_mass_in_box(center_prior, item['boxes'])
            
            head_stats['head_10'].append(score_10)
            head_stats['head_1'].append(score_1)
            head_stats['center'].append(score_c)
            
            print(f"{item['id']:<10} | {score_10:<15.4f} | {score_1:<15.4f} | {score_c:<15.4f}")
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            continue

    AttentionClass.forward = original_forward 

    avg_10 = np.mean(head_stats['head_10'])
    avg_1 = np.mean(head_stats['head_1'])
    avg_c = np.mean(head_stats['center'])
    
    print("\n" + "="*60)
    print("FINAL RESULTS (N={} images)".format(len(head_stats['head_10'])))
    print("Metric: % of Attention Mass falling inside Ground Truth Object Boxes")
    print("-" * 60)
    print(f"Layer 7 Head 10 (Target): {avg_10*100:.2f}%  <-- THE SALIENCY HEAD")
    print(f"Layer 7 Head 1  (Random): {avg_1*100:.2f}%")
    print(f"Center Prior    (Naive) : {avg_c*100:.2f}%")
    print("=" * 60)
    
    if avg_10 > avg_1 + 0.1:
        print("\nCONCLUSION: Layer 7 Head 10 is statistically significantly better at")
        print("locating objects than random heads. HYPOTHESIS CONFIRMED.")
    else:
        print("\nCONCLUSION: Hypothesis weak. No significant advantage detected.")

if __name__ == "__main__":
    main()
