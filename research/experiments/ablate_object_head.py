import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import open_clip

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
    if not os.path.exists(val_dir):
         val_dir = "../data/val2017"
    
    target_file = "000000000139.jpg"
    path = os.path.join(val_dir, target_file)
    print(f"Analyzing {path}...")
    
    transform = get_transform()
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = "table"
    text_tokens = tokenizer([text])[0] 
    target_token_id = text_tokens[1].item()
    print(f"Target token '{text}' ID: {target_token_id}")

    def run_logit_lens(experiment_name="Baseline"):
        print(f"\n--- Running Logit Lens: {experiment_name} ---")
        activations = {}
        def get_hook(i):
            def hook(module, input, output):
                activations[i] = output.detach()
            return hook
        
        hooks = []
        for i, block in enumerate(model.visual.transformer.resblocks):
            hooks.append(block.register_forward_hook(get_hook(i)))
            
        with torch.no_grad():
            model.visual(tensor)
            
        for h in hooks: h.remove()
        
        results = []
        layer_indices = sorted(activations.keys())
        
        proj = model.visual.proj
        ln_post = model.visual.ln_post
        super_head = model.super_head 
        
        print(f"{'Layer':<10} {'Rank':<10} {'Top Token':<20} {'Score':<10}")
        
        for i in layer_indices:
            x = activations[i][:, 0, :] 
            
            x = ln_post(x)
            if proj is not None:
                x = x @ proj
                
            logits = super_head(x)[0]
            
            target_score = logits[target_token_id].item()
            rank = (logits > target_score).sum().item() + 1
            
            top_val, top_idx = logits.topk(1)
            top_token = tokenizer.decode([top_idx.item()]).strip()
            
            print(f"{i+1:<10} {rank:<10} {top_token:<20} {target_score:.2f}")
            results.append((rank, top_token))
            
        return results

    baseline_results = run_logit_lens("Baseline")
    baseline_ranks = [r[0] for r in baseline_results]
    
    layer_idx = 6
    head_idx = 10
    head_dim = 64
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim
    
    print(f"\nAblating Layer {layer_idx+1}, Head {head_idx+1} (indices {start_idx}:{end_idx})...")
    
    attn_module = model.visual.transformer.resblocks[layer_idx].attn
    
    original_weights = attn_module.out_proj.weight.data.clone()
    
    attn_module.out_proj.weight.data[:, start_idx:end_idx] = 0.0
    
    ablated_results = run_logit_lens("Ablated HEAD 10")
    ablated_ranks = [r[0] for r in ablated_results]
    
    print("\nResults (Rank of 'table'):")
    print(f"{'Layer':<10} {'Baseline':<10} {'Ablated':<10} {'Delta':<10}")
    for i in range(len(baseline_ranks)):
        base = baseline_ranks[i]
        abl = ablated_ranks[i]
        delta = abl - base
        print(f"{i+1:<10} {base:<10} {abl:<10} {delta:<10}")

    plt.figure(figsize=(10, 6))
    layers = range(1, len(baseline_ranks) + 1)
    
    plt.semilogy(layers, baseline_ranks, marker='o', label='Baseline')
    plt.semilogy(layers, ablated_ranks, marker='x', linestyle='--', label='Ablated (Head 10)')
    
    plt.title("Impact of Ablating Layer 7 Head 10 on 'Table' Rank")
    plt.xlabel("Layer")
    plt.ylabel("Rank (Log Scale) - Lower is Better")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    plt.gca().invert_yaxis()
    
    os.makedirs("../plots", exist_ok=True)
    plt.savefig("../plots/ablation_rank_result.png")
    print("\nSaved plot to plots/ablation_rank_result.png")

if __name__ == "__main__":
    main()
