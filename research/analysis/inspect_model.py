import sys
import os
import torch
import open_clip
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from train import SuperCLIP

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing SuperCLIP model structure...")
    model = SuperCLIP(base_model='ViT-B-32', vocab_size=49408)
    
    checkpoint_path = "../../logs/checkpoints/epoch_20.pt"
    if not os.path.exists(checkpoint_path): checkpoint_path = "../logs/checkpoints/epoch_20.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    print("\nVisual Encoder Structure:")
    if hasattr(model.visual, 'transformer'):
        print(f"Blocks found in model.visual.transformer.resblocks: {len(model.visual.transformer.resblocks)} layers")
    else:
        print("Could not find standard 'transformer.resblocks'. Printing keys of visual model:")
        print(model.visual)
        
    print(f"\nSuper Head: {model.super_head}")
    
if __name__ == "__main__":
    main()
