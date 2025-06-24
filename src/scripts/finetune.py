#!/usr/bin/env python3
import argparse
import os
import yaml
from dataclasses import dataclass
from configs.base_config import BaseConfig
from models.gmae import GMAE, GMAEConfig, GMAEForClassification
from data.datasets import CustomDataset
from data.transforms import build_finetune_transforms
from engine.trainer import FinetuneTrainer
from utils.logger import init_wandb
import torch
from torch.utils.data import DataLoader


@dataclass
class FineTuneConfig(BaseConfig):
    num_classes: int = 100
    freeze_encoder_epochs: int = 5
    pretrained_checkpoint: str = None


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
        mlp_ratio=config.mlp_ratio,
        dilation_rates=config.dilation_rates,
        ffn_kernel_sizes=config.ffn_kernel_sizes
    )
    
    pretrained_model = GMAE(model_config)
    if config.pretrained_checkpoint:
        pretrained_model.load_state_dict(torch.load(config.pretrained_checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])

    model = GMAEForClassification(pretrained_model.encoder, config.num_classes, model_config).to(device)
    
    for param in model.encoder.parameters():
        param.requires_grad = False

    train_transforms = build_finetune_transforms(config.img_size, is_train=True)
    val_transforms = build_finetune_transforms(config.img_size, is_train=False)

    train_dataset = CustomDataset(config.data_path, transform=train_transforms, num_classes=config.num_classes)
    val_dataset = CustomDataset(config.data_path, transform=val_transforms, num_classes=config.num_classes)
    
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
    
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMAE Fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = BaseConfig(**config_dict)
    main(config)