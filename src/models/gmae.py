import torch
import torch.nn as nn
from dataclasses import dataclass, field
from .modules import OverlapPatchEmbed, GMSTBlock



@dataclass
class GMAEConfig:
    in_channels: int = 3
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    dilation_rates: list = field(default_factory=lambda: [1, 3, 5])
    ffn_kernel_sizes: list = field(default_factory=lambda: [3, 5, 7])
    overlap_patches: bool = False

    def __post_init__(self):
        self.ffn_hidden_channels_encoder = self._adjust_dim(self.embed_dim)
        self.ffn_hidden_channels_decoder = self._adjust_dim(self.decoder_embed_dim)
    
    def _adjust_dim(self, dim):
        value = int(dim * self.mlp_ratio)
        return value + (3 - (value % 3)) if value % 3 != 0 else value



class GMSTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(
            config.in_channels, config.embed_dim, config.patch_size, config.overlap_patches)
        self.blocks = nn.ModuleList([
            GMSTBlock(config.embed_dim, config.num_heads, config.ffn_hidden_channels_encoder)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.config = config
        self.num_patches = self._calculate_num_patches(config.img_size)

    def _calculate_num_patches(self, img_size):
        H = self.patch_embed.calculate_output_size(img_size)
        W = self.patch_embed.calculate_output_size(img_size)
        return H * W

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        return self.norm(x), H, W



class GMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = GMSTEncoder(config)
        self.num_patches = self.encoder.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            GMSTBlock(config.decoder_embed_dim, config.decoder_num_heads, config.ffn_hidden_channels_decoder)
            for _ in range(config.decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(config.decoder_embed_dim, config.patch_size**2 * config.in_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.config.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * imgs.shape[1])

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs):
        x, H, W = self.encoder(imgs)
        x = x + self.pos_embed
        x_visible, mask, ids_restore = self.random_masking(x, self.config.mask_ratio)
        return x_visible, mask, ids_restore, H, W

    def forward_decoder(self, x, ids_restore, H, W):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x, H, W)
        return self.decoder_pred(self.decoder_norm(x))

    def forward(self, imgs):
        latent, mask, ids_restore, H, W = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore, H, W)
        target = self.patchify(imgs)
        mask_exp = mask.unsqueeze(-1).repeat(1, 1, target.shape[-1])
        squared_diff = (pred - target)**2
        mask_sum = mask_exp.sum().clamp(min=1e-6)
        loss = (squared_diff * mask_exp).sum() / mask_sum
        return loss



class GMAEForClassification(nn.Module):
    def __init__(self, encoder, num_classes, config):
        super().__init__()
        self.encoder = encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, (config.img_size//config.patch_size)**2, config.embed_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim), 
            nn.Linear(config.embed_dim, num_classes)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, imgs):
        x, H, W = self.encoder.patch_embed(imgs)
        x = x + self.pos_embed
        for blk in self.encoder.blocks:
            x = blk(x, H, W)
        return self.classifier(self.encoder.norm(x).mean(dim=1))