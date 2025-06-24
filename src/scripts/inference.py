import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

from models.gmae import GMAE, GMAEConfig
from configs.base_config import BaseConfig
from data.datasets import CustomDataset
from data.transforms import build_inference_transforms



def parse_args():
    parser = argparse.ArgumentParser(description="GMAE Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    return parser.parse_args()


def save_reconstructions(args, original, reconstructed, output_dir, batch_idx):
    os.makedirs(os.path.join(output_dir, "reconstructions"), exist_ok=True)
    
    for i in range(min(len(original), args.num_samples)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        orig_img = original[i].permute(1, 2, 0).cpu().numpy()
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        recon_img = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
        axes[1].imshow(recon_img)
        axes[1].set_title("Reconstructed")
        axes[1].axis('off')
        
        plt.savefig(os.path.join(output_dir, "reconstructions", f"compare_b{batch_idx}_s{i}.png"))
        plt.close()



def save_metrics(metrics, output_dir):
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    with open(os.path.join(output_dir, "metrics", "classification_report.txt"), "w") as f:
        f.write(metrics['classification_report'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "metrics", "confusion_matrix.png"))
    plt.close()



def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    config = BaseConfig(**config_data)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    model_config = GMAEConfig(
        in_channels=config.in_channels,
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        mask_ratio=config.mask_ratio,
        dilation_rates=config.dilation_rates,
        ffn_kernel_sizes=config.ffn_kernel_sizes,
        overlap_patches=config.overlap_patches
    )
    model = GMAE(model_config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_transforms = build_inference_transforms(config.img_size, is_train=False)
    val_dataset = CustomDataset(
        root_dir=args.data_dir, 
        transform=val_transforms, 
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"Starting inference on {len(val_loader.dataset)} samples...")
    os.makedirs(args.output_dir, exist_ok=True)

    all_outputs = []
    all_targets = []
    all_losses = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Inference")):
            try:
                images = images.to(device)
                targets = targets.to(device) if targets is not None else None
                
                outputs = model(images)
                
                batch_output = {
                    'inputs': images.cpu(),
                    'outputs': outputs.cpu(),
                    'targets': targets.cpu() if targets is not None else None
                }
                torch.save(batch_output, os.path.join(args.output_dir, f"batch_{batch_idx}.pt"))
                
                if batch_idx == 0 and args.num_samples > 0:
                    save_reconstructions(args, images, outputs, args.output_dir, batch_idx)
                
                if targets is not None:
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue


    if len(all_targets) > 0:
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        if len(all_outputs.shape) == 2:
            preds = np.argmax(all_outputs, axis=1)
            metrics = {
                'classification_report': classification_report(all_targets, preds),
                'confusion_matrix': confusion_matrix(all_targets, preds)
            }
            save_metrics(metrics, args.output_dir)
            
            print("\nClassification Report:")
            print(metrics['classification_report'])
        
        else:  
            mse_loss = np.mean((all_outputs - all_targets) ** 2)
            print(f"\nReconstruction MSE: {mse_loss:.4f}")

    print(f"\nInference completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()