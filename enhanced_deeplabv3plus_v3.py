import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -----------------------------
# ASPP Module
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        g = self.global_pool(x)
        g = self.global_conv(g)
        g = F.interpolate(g, size=size, mode='bilinear', align_corners=False)

        fused = torch.cat([b1, b2, b3, b4, g], dim=1)
        return self.project(fused)


# -----------------------------
# Attention-Guided Feature Fusion (AGFF)
# -----------------------------
class AttentionFusion(nn.Module):
    """
    Fuse high-level (ASPP upsampled) and low-level features with
    channel + spatial attention.
    """
    def __init__(self, ch_high=256, ch_low=48, ch_fused=256, reduction=16):
        super().__init__()

        self.proj_low = nn.Conv2d(ch_low, ch_fused, 1, bias=False)

        # Channel attention (like SE)
        self.ca_fc1 = nn.Linear(ch_fused * 2, (ch_fused * 2) // reduction)
        self.ca_fc2 = nn.Linear((ch_fused * 2) // reduction, ch_fused * 2)

        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # Final fusion conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(ch_fused * 2, ch_fused, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_fused),
            nn.ReLU(inplace=True),
        )

    def forward(self, high, low):
        # high: (B, 256, H, W)
        # low:  (B, 48,  H, W) -> project to 256
        low_proj = self.proj_low(low)

        # concat for attention
        concat = torch.cat([high, low_proj], dim=1)  # (B, 512, H, W)

        # ----- Channel attention -----
        b, c, h, w = concat.shape
        gp = F.adaptive_avg_pool2d(concat, (1, 1)).view(b, c)  # (B, 512)
        ca = self.ca_fc1(gp)
        ca = F.relu(ca, inplace=True)
        ca = self.ca_fc2(ca)
        ca = torch.sigmoid(ca).view(b, c, 1, 1)  # (B, 512, 1, 1)

        concat_ca = concat * ca

        # ----- Spatial attention -----
        avg_sp = torch.mean(concat_ca, dim=1, keepdim=True)
        max_sp, _ = torch.max(concat_ca, dim=1, keepdim=True)
        sp = torch.cat([avg_sp, max_sp], dim=1)  # (B, 2, H, W)
        sa = torch.sigmoid(self.sa_conv(sp))     # (B, 1, H, W)

        concat_casa = concat_ca * sa

        # ----- Final fusion -----
        fused = self.fuse_conv(concat_casa)  # (B, 256, H, W)
        return fused


# -----------------------------
# Boundary Refinement Module (BRM)
# -----------------------------
class BoundaryRefinement(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        out = self.block(x)
        out = F.relu(out + x, inplace=True)
        return out


# -----------------------------
# Enhanced DeepLabV3+ V3
# -----------------------------
class DeepLabV3PlusV3(nn.Module):
    """
    DeepLabV3+ V3:
    - ResNet50 backbone
    - ASPP on high-level
    - Attention-guided fusion of ASPP & low-level
    - Dual decoders: segmentation path + horizon path
    - Boundary refinement before segmentation classifier
    - Auxiliary segmentation head on high-level

    Returns:
      {
        "main":    (B, num_classes, H, W),
        "aux":     (B, num_classes, H, W),
        "horizon": (B, horizon_bins)
      }
    """
    def __init__(self, num_classes=3, horizon_bins=32, pretrained=True):
        super().__init__()

        # Backbone
        try:
            base = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            base = models.resnet50(pretrained=pretrained)

        self.backbone = nn.Sequential(
            base.conv1,   # 0
            base.bn1,     # 1
            base.relu,    # 2
            base.maxpool, # 3
            base.layer1,  # 4 -> low-level
            base.layer2,  # 5
            base.layer3,  # 6
            base.layer4,  # 7 -> high-level
        )

        low_in = 256
        high_in = 2048

        # Low-level projection
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_in, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ASPP
        self.aspp = ASPP(high_in, 256)

        # Attention-guided fusion
        self.fusion = AttentionFusion(ch_high=256, ch_low=48, ch_fused=256, reduction=16)

        # Shared fused feature (for both decoders)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Segmentation-specific refinement + classifier
        self.seg_brm = BoundaryRefinement(channels=256)
        self.seg_classifier = nn.Conv2d(256, num_classes, 1)

        # Auxiliary segmentation head (from high-level)
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(high_in, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        # Horizon-specific decoder: lightweight
        self.horizon_decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.horizon_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, horizon_bins),
        )

    def forward(self, x):
        input_size = x.shape[2:]

        cur = x
        low = None
        for i, layer in enumerate(self.backbone):
            cur = layer(cur)
            if i == 4:
                low = cur  # low-level
        high = cur  # high-level

        # ASPP on high
        aspp_out = self.aspp(high)  # (B, 256, H/32, W/32)

        # Low projection
        low_proj = self.low_proj(low)  # (B, 48, H/4, W/4)

        # Upsample ASPP to low-level resolution
        aspp_up = F.interpolate(aspp_out, size=low_proj.shape[2:], mode='bilinear', align_corners=False)

        # Attention-guided fusion
        fused = self.fusion(aspp_up, low_proj)  # (B, 256, H/4, W/4)

        # Shared decoder
        shared = self.shared_decoder(fused)  # (B, 256, H/4, W/4)

        # Segmentation path
        seg_feat = self.seg_brm(shared)      # boundary refined
        seg_logits = self.seg_classifier(seg_feat)  # (B, C, H/4, W/4)
        seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)

        # Auxiliary seg from high-level
        aux_logits = self.aux_classifier(high)
        aux_logits = F.interpolate(aux_logits, size=input_size, mode='bilinear', align_corners=False)

        # Horizon path (dual decoder)
        hor_feat = self.horizon_decoder(shared)  # (B, 64, H/4, W/4)
        hor_pooled = F.adaptive_avg_pool2d(hor_feat, (1, 1)).view(hor_feat.size(0), -1)  # (B, 64)
        horizon_logits = self.horizon_fc(hor_pooled)  # (B, horizon_bins)

        return {
            "main": seg_logits,
            "aux": aux_logits,
            "horizon": horizon_logits,
        }
