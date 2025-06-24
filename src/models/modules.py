import torch
import torch.nn as nn
import torch.nn.functional as F



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, overlap=True):
        super().__init__()
        stride = patch_size // 2 if overlap else patch_size
        padding = patch_size // 2 if overlap else 0
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W

    def calculate_output_size(self, input_size):
        return (input_size + 2 * self.padding - self.patch_size) // self.stride + 1



class MSDC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )
        self.dconv1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=3, dilation=3),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.se_fc1 = nn.Linear(dim, dim // 16)
        self.se_fc2 = nn.Linear(dim // 16, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        f0 = self.conv1x1(x)
        f1 = self.dconv1(x)
        f2 = self.dconv2(x)
        f3 = self.dconv3(x)
        f_concat = torch.cat([f0, f1, f2, f3], dim=1)
        msff = self.fusion(f_concat)
        avg_pool = torch.mean(msff, dim=1, keepdim=True)
        max_pool = torch.max(msff, dim=1, keepdim=True)[0]
        sa_concat = torch.cat([avg_pool, max_pool], dim=1)
        sa_mask = torch.sigmoid(self.sa_conv(sa_concat))
        sa_output = msff * sa_mask
        se_squeeze = F.adaptive_avg_pool2d(sa_output, (1, 1)).flatten(1)
        se_excitation = torch.sigmoid(self.se_fc2(F.gelu(self.se_fc1(se_squeeze))))
        return sa_output * se_excitation.view(-1, self.se_fc2.out_features, 1, 1) + identity



class MSC_FFN(nn.Module):
    def __init__(self, dim, hidden, out):
        super().__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv2d(dim, hidden // 3, 3, padding=1, dilation=1),
            nn.BatchNorm2d(hidden // 3),
            nn.GELU()
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(dim, hidden // 3, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden // 3),
            nn.GELU()
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(dim, hidden // 3, 3, padding=3, dilation=3),
            nn.BatchNorm2d(hidden // 3),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, out)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        f1 = self.dconv1(x)
        f2 = self.dconv2(x)
        f3 = self.dconv3(x)
        f_concat = torch.cat([f1, f2, f3], dim=1)
        msf = self.fusion(f_concat)
        msf_flat = msf.flatten(2).permute(0, 2, 1)
        output = F.gelu(self.fc1(msf_flat))
        return self.fc2(output)



class GMSTBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.msdc = MSDC(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mscffn = MSC_FFN(dim, ffn_dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))
        self.gamma3 = nn.Parameter(torch.ones(dim))
        nn.init.normal_(self.gamma1, mean=1.0, std=0.02)
        nn.init.normal_(self.gamma2, mean=1.0, std=0.02)
        nn.init.normal_(self.gamma3, mean=1.0, std=0.02)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_img = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.gamma1 * self.msdc(x_img).flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.gamma2 * attn_out
        x_img2 = x.permute(0, 2, 1).view(B, C, H, W)
        return self.norm3(x + self.gamma3 * self.mscffn(x_img2))
