import torch
import torch.nn as nn
import open_clip
from PIL import Image
from tqdm import tqdm
import os
import json
import tarfile
import glob

# --- Vanilla CLIP (With no extra Super Head) ---
class VanillaCLIP(nn.Module):
    def __init__(self, base_model='ViT-B-32'):
        super().__init__()
        self.clip = open_clip.create_model(base_model, pretrained='openai')
        self.visual = self.clip.visual
        self.text = self.clip.transformer
        self.token_embedding = self.clip.token_embedding
        self.ln_final = self.clip.ln_final
        self.logit_scale = self.clip.logit_scale

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text):
        return self.clip.encode_text(text)

def run_eval(checkpoint_path, val_tar_path="data/shards/val.tar", data_dir="data/val2017", ann_file="data/annotations/captions_val2017.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Baseline Checkpoint: {checkpoint_path}...")

    model = VanillaCLIP().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    # Preprocessing
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Load Annotations
    with open(ann_file, 'r') as f:
        data = json.load(f)

    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    img_id_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_captions: img_id_to_captions[img_id] = []
        img_id_to_captions[img_id].append(ann['caption'])

    # Get IDs from VAL TAR
    test_ids = []
    with tarfile.open(val_tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".txt"): 
                fid = os.path.basename(member.name).replace(".txt", "")
                if fid.isdigit(): test_ids.append(int(fid))
    
    test_ids = list(set(test_ids))[:1000] # Same subset
    print(f"Running Baseline Eval on {len(test_ids)} held-out images...")

    img_embeds = []
    txt_embeds = []

    with torch.no_grad():
        for img_id in tqdm(test_ids):
            filename = img_id_to_filename[img_id]
            caption = img_id_to_captions[img_id][0]
            
            path = os.path.join(data_dir, filename)
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            text = tokenizer([caption]).to(device)

            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)
            
            img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True)

            img_embeds.append(img_feat.cpu())
            txt_embeds.append(txt_feat.cpu())

    img_embeds = torch.cat(img_embeds, dim=0)
    txt_embeds = torch.cat(txt_embeds, dim=0)
    
    sim_matrix = img_embeds @ txt_embeds.t()
    labels = torch.arange(len(test_ids)).to(sim_matrix.device).view(-1, 1)
    
    r1 = (sim_matrix.topk(1, dim=1).indices == labels).sum().item() / len(test_ids)
    r5 = (sim_matrix.topk(5, dim=1).indices == labels).sum().item() / len(test_ids)
    r10 = (sim_matrix.topk(10, dim=1).indices == labels).sum().item() / len(test_ids)

    print(f"\n--- Baseline (Vanilla CLIP) Results ---")
    print(f"Recall@1:  {r1*100:.2f}%")
    print(f"Recall@5:  {r5*100:.2f}%")
    print(f"Recall@10: {r10*100:.2f}%")

if __name__ == "__main__":
    # Finds the latest baseline checkpoint
    checkpoints = sorted(glob.glob("logs/checkpoints_baseline/*.pt"))
    if checkpoints:
        run_eval(checkpoints[-1])
    else:
        print("No baseline checkpoints found.")