from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BaseConfig:
    seed: int = 42
    device: str = "cuda"
    log_to_wandb: bool = True
    project_name: str = "GMAE"
    entity: str = None
    run_name: str = "gmae-experiment"
    save_dir: str = "/home/student/projects/gmae/checkpoints"
    
    data_path: str = "/home/student/projects/gmae/data"
    pretrained_checkpoint: Optional[str] = None
    in_channels: int = 3
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    num_classes: Optional[int] = None
    
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.6
    dilation_rates: List[int] = field(default_factory=lambda: [1, 3, 5])
    ffn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    overlap_patches: bool = True
    freeze_encoder_epochs: Optional[int] = 5
    
    epochs: int = 500
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    save_freq: int = 50
    fp16: bool = True