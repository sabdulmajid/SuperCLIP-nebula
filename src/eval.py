import torch
import torch.nn as nn
import open_clip
from PIL import Image
from tqdm import tqdm
import os
import json
import tarfile

class SuperCLIP(nn.Module):
    def __init__(self, base_model='ViT-B-32', vocab_size=49408):
        super().__init__()
        self.clip = open_clip.create_model(base_model, pretrained='openai')
        self.visual = self.clip.visual
        self.text = self.clip.transformer
        self.token_embedding = self.clip.token_embedding
        self.ln_final = self.clip.ln_final
        self.logit_scale = self.clip.logit_scale
        visual_dim = self.visual.output_dim
        self.super_head = nn.Linear(visual_dim, vocab_size)

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text):
        return self.clip.encode_text(text)

def run_eval(checkpoint_path, val_tar_path="data/shards/val.tar", data_dir="data/val2017", ann_file="data/annotations/captions_val2017.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {checkpoint_path}...")

    model = SuperCLIP().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Fix DataParallel keys
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
    print("Loading COCO annotations...")
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # Map ImageID -> Filename & Captions
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    img_id_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        img_id_to_captions[img_id].append(ann['caption'])

    # Get IDs from the Validation TAR
    print(f"Reading validation IDs from {val_tar_path}...")
    test_ids = []
    
    with tarfile.open(val_tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".txt"): 
                # Extract ID (remove extension)
                file_id = os.path.basename(member.name).replace(".txt", "")
                if file_id.isdigit():
                    test_ids.append(int(file_id))
    
    # Remove duplicates if any (tar might have multiple files per ID)
    test_ids = list(set(test_ids))
    
    # Limit to 1000 for speed (Randomly sampled from the Val set)
    # If Val set < 1000, take all of them.
    test_ids = test_ids[:1000]
    
    print(f"STRICT EVALUATION: Testing on {len(test_ids)} held-out images.")

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

    # Similarity Matrix
    print("Computing Similarity Matrix...")
    sim_matrix = img_embeds @ txt_embeds.t()

    num_samples = len(test_ids)
    labels = torch.arange(num_samples).to(sim_matrix.device).view(-1, 1) # Correct broadcasting
    
    top1 = sim_matrix.topk(1, dim=1).indices
    top5 = sim_matrix.topk(5, dim=1).indices
    top10 = sim_matrix.topk(10, dim=1).indices

    r1 = (top1 == labels).sum().item() / num_samples
    r5 = (top5 == labels).sum().item() / num_samples
    r10 = (top10 == labels).sum().item() / num_samples

    print(f"\n--- SuperCLIP Results (Held-Out Data) ---")
    print(f"Recall@1:  {r1*100:.2f}%")
    print(f"Recall@5:  {r5*100:.2f}%")
    print(f"Recall@10: {r10*100:.2f}%")

if __name__ == "__main__":
    # Ensure this points to the checkpoint you want to test
    # (e.g., epoch_20.pt if you let it run long enough)
    # If training is still running, check what is available in logs/checkpoints/
    import glob
    checkpoints = sorted(glob.glob("logs/checkpoints/*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        run_eval(latest)
    else:
        print("No checkpoints found yet!")