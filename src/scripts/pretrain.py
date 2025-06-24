#!/usr/bin/env python3
import argparse
import os
import yaml
from configs.base_config import BaseConfig
from models.gmae import GMAE, GMAEConfig
from data.datasets import CustomDataset
from data.transforms import build_pretrain_transforms
from engine.trainer import PreTrainer
from utils.logger import init_wandb
import torch
from torch.utils.data import DataLoader



def main(config):
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
    
    train_transforms = build_pretrain_transforms(config.img_size, is_train=True)
    val_transforms = build_pretrain_transforms(config.img_size, is_train=False)
    
    train_dataset = CustomDataset(
        root_dir=config.data_path, 
        transform=train_transforms, 
        mode='train'
    )
    val_dataset = CustomDataset(
        root_dir=config.data_path, 
        transform=val_transforms, 
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print("Train dataset size:", len(train_loader.dataset))
    print("Validation dataset size:", len(val_loader.dataset))
    
    init_wandb(config)
    os.makedirs(config.save_dir, exist_ok=True)
    
    trainer = PreTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMAE Pretraining")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = BaseConfig(**config_dict)
    main(config)