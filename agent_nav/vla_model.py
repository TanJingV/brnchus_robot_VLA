from typing import List, Sequence, Union

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTransformerEncoder(nn.Module):
    # FT-Transformer 风格状态编码器
    def __init__(self, state_dim: int, d_model: int = 256, nhead: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.scalar_proj = nn.Linear(1, d_model)
        self.feature_embed = nn.Parameter(torch.randn(1, state_dim, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: [B, F]
        x = state.unsqueeze(-1)  # [B, F, 1]
        x = self.scalar_proj(x) + self.feature_embed
        cls = self.cls.expand(state.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0, :])  # cls token


class VLAFoundationPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        clip_model_name: str = "ViT-H-14",
        clip_pretrained: str = "laion2b_s32b_b79k",
        train_clip: bool = False,
        state_width: int = 256,
        fusion_width: int = 1024,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained

        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.clip_embed_dim = int(getattr(self.clip, "text_projection").shape[1])

        if not train_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        self.state_encoder = StateTransformerEncoder(state_dim=state_dim, d_model=state_width)

        self.visual_proj = nn.Linear(self.clip_embed_dim, fusion_width)
        self.text_proj = nn.Linear(self.clip_embed_dim, fusion_width)
        self.state_proj = nn.Linear(state_width, fusion_width)

        self.fusion_norm = nn.LayerNorm(fusion_width * 3)
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_width * 3, fusion_width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_width * 2, fusion_width),
            nn.GELU(),
            nn.Linear(fusion_width, 3),
            nn.Tanh(),
        )

        self.register_buffer("img_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def _prepare_text(self, texts: Union[Sequence[str], torch.Tensor], device: torch.device) -> torch.Tensor:
        if isinstance(texts, torch.Tensor):
            return texts.to(device)
        return self.tokenizer(list(texts)).to(device)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B,3,H,W] in [0,1]
        target_size = int(getattr(self.clip.visual, "image_size", 224))
        if image.shape[-1] != target_size or image.shape[-2] != target_size:
            image = F.interpolate(image, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return (image - self.img_mean) / self.img_std

    def forward(self, image: torch.Tensor, state: torch.Tensor, texts: Union[Sequence[str], torch.Tensor]) -> torch.Tensor:
        image = self._prepare_image(image)
        text_tokens = self._prepare_text(texts, image.device)

        v = self.clip.encode_image(image).float()
        t = self.clip.encode_text(text_tokens).float()
        s = self.state_encoder(state)

        v = self.visual_proj(v)
        t = self.text_proj(t)
        s = self.state_proj(s)

        fused = self.fusion_norm(torch.cat([v, s, t], dim=1))
        return self.policy_head(fused)


def build_vla(
    state_dim: int,
    clip_model_name: str = "ViT-H-14",
    clip_pretrained: str = "laion2b_s32b_b79k",
    train_clip: bool = False,
    state_width: int = 256,
    fusion_width: int = 1024,
) -> VLAFoundationPolicy:
    return VLAFoundationPolicy(
        state_dim=state_dim,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        train_clip=train_clip,
        state_width=state_width,
        fusion_width=fusion_width,
    )

