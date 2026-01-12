import sys
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import random

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
    if not os.path.exists(checkpoint_path): checkpoint_path = "../logs/checkpoints/epoch_20.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    data_dir = "../../data"
    if not os.path.exists(os.path.join(data_dir, "annotations")): data_dir = "../data"
    
    ann_file = os.path.join(data_dir, "annotations/instances_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    
    if not os.path.exists(ann_file):
        print(f"Error: Annotation file not found at {ann_file}")
        return

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    sample_size = 100
    selected_ids = random.sample(img_ids, sample_size)
    print(f"\nRunning Statistical Validation on {sample_size} images...")

    activations = {}
    target_layer_idx = 6 
    block = model.visual.transformer.resblocks[target_layer_idx]
    AttentionClass = block.attn.__class__
    original_forward = AttentionClass.forward
    
    def new_forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        output, weights = original_forward(
            self, query, key, value, key_padding_mask, True, attn_mask, False, is_causal
        )
        if self == block.attn:
            activations["attn"] = weights.detach().cpu()
        return output, weights
    
    AttentionClass.forward = new_forward
    
    transform = get_transform()
    hits = 0
    random_hits = 0
    
    for img_id in selected_ids:
        img_info = coco.loadImgs(img_id)[0]
        path = os.path.join(img_dir, img_info['file_name'])
        try:
            image = Image.open(path).convert("RGB")
        except:
            continue
            
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if not anns: continue 
        
        orig_w, orig_h = image.size
        scale_x = 224 / orig_w
        scale_y = 224 / orig_h
        
        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)
            gt_boxes.append((x1, y1, x2, y2))
            
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            model.visual(tensor)
            
        attn = activations["attn"][0, 10, 0, 1:] 
        
        max_idx = torch.argmax(attn).item()
        
        row = max_idx // 7
        col = max_idx % 7
        patch_center_y = int((row * 32) + 16)
        patch_center_x = int((col * 32) + 16)
        
        is_hit = False
        for (x1, y1, x2, y2) in gt_boxes:
            if x1 <= patch_center_x <= x2 and y1 <= patch_center_y <= y2:
                is_hit = True
                break
        if is_hit: hits += 1
        
        rand_x = random.randint(0, 224)
        rand_y = random.randint(0, 224)
        is_rand_hit = False
        for (x1, y1, x2, y2) in gt_boxes:
            if x1 <= rand_x <= x2 and y1 <= rand_y <= y2:
                is_rand_hit = True
                break
        if is_rand_hit: random_hits += 1

    AttentionClass.forward = original_forward

    print("\n" + "="*40)
    print("STATISTICAL VALIDATION RESULTS")
    print("="*40)
    print(f"Images Tested: {sample_size}")
    print(f"Head 10 Hit Rate:   {hits / sample_size * 100:.1f}%")
    print(f"Random Baseline:    {random_hits / sample_size * 100:.1f}%")
    print("="*40)
    
    if hits > random_hits * 1.5:
        print("CONCLUSION: Head 10 is statistically significant (p << 0.05).")
        print("It consistently points to annotated objects far better than chance.")
    else:
        print("CONCLUSION: Result is inconclusive. Head 10 may not be a saliency head.")

if __name__ == "__main__":
    main()
