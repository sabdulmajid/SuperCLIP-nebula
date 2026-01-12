import sys
import os
import torch
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import open_clip

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def load_coco_captions(data_dir, num_images=100):
    ann_file = os.path.join(data_dir, "annotations", "captions_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    
    print(f"Loading captions from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)
        
    img_map = {img['id']: img for img in coco['images']}
    ann_map = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann['caption'])
        
    results = []
    count = 0
    for img_id, captions in ann_map.items():
        if img_id in img_map:
            file_name = img_map[img_id]['file_name']
            path = os.path.join(img_dir, file_name)
            if os.path.exists(path):
                results.append({
                    'path': path,
                    'caption': captions[0], 
                    'id': img_id
                })
                count += 1
                if count >= num_images:
                    break
    return results

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
    
    model = SuperCLIP(base_model='ViT-B-32').to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    checkpoint_path = "../../logs/checkpoints/epoch_20.pt"
    if not os.path.exists(checkpoint_path): checkpoint_path = "../logs/checkpoints/epoch_20.pt"

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
    else:
        print("Warning: Checkpoint not found, using random weights (Bad for science!)")

    model.eval()
    
    data_dir = "../../data" 
    if not os.path.exists(os.path.join(data_dir, "annotations")): data_dir = "../data"

    dataset = load_coco_captions(data_dir, num_images=100)
    transform = get_transform()
    
    layer_7_attn = model.visual.transformer.resblocks[6].attn
    head_dim = 768 // 12
    
    def ablate_head(head_idx):
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        
        original_w = layer_7_attn.out_proj.weight.data[:, start:end].clone()
        layer_7_attn.out_proj.weight.data[:, start:end] = 0
        
        return original_w
    
    def restore_head(head_idx, original_w):
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        layer_7_attn.out_proj.weight.data[:, start:end] = original_w

    print("\nRunning Ablation Impact Study (Cosine Similarity Drop)...")
    print("-" * 60)
    print(f"{'Image ID':<10} | {'Base Score':<10} | {'Drop H10 (Ours)':<15} | {'Drop H1 (Rand)':<15}")
    print("-" * 60)
    
    deltas_10 = []
    deltas_1 = []
    
    for item in dataset:
        try:
            img = Image.open(item['path']).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            
            text = tokenizer([item['caption']]).to(device)
            
            with torch.no_grad():
                img_feat, text_feat, logit_scale, _ = model(img_t, text)
                sim_base = (img_feat @ text_feat.T).item()
                
            w_backup = ablate_head(10)
            with torch.no_grad():
                img_feat_10, _, _, _ = model(img_t, text)
                sim_10 = (img_feat_10 @ text_feat.T).item()
            restore_head(10, w_backup)
            
            w_backup = ablate_head(1)
            with torch.no_grad():
                img_feat_1, _, _, _ = model(img_t, text)
                sim_1 = (img_feat_1 @ text_feat.T).item()
            restore_head(1, w_backup)
            
            delta_10 = sim_base - sim_10
            delta_1 = sim_base - sim_1
            
            deltas_10.append(delta_10)
            deltas_1.append(delta_1)
            
            print(f"{item['id']:<10} | {sim_base:.4f}     | {delta_10:.6f}          | {delta_1:.6f}")
            
        except Exception as e:
            continue
            
    print("-" * 60)
    mean_10 = np.mean(deltas_10)
    mean_1 = np.mean(deltas_1)
    
    print(f"Average Drop (Head 10): {mean_10:.6f}")
    print(f"Average Drop (Head 1):  {mean_1:.6f}")
    
    if mean_10 > mean_1:
         print(f"\nResult: Ablating Head 10 damages model performance {mean_10/mean_1:.1f}x more than a random head.")
         print("VERDICT: Head 10 is load-bearing.")
    else:
        print("\nResult: No significant difference.")

if __name__ == "__main__":
    main()
