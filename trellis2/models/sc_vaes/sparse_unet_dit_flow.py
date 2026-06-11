from typing import *
from functools import partial
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import sparse as sp
from ...modules.transformer import AbsolutePositionEmbedder
from ...modules.sparse.transformer import ModulatedSparseTransformerBlock
from ...modules.utils import convert_module_to, convert_module_to_f16, convert_module_to_f32, manual_cast, str_to_dtype, zero_module
from ...modules.norm import LayerNorm32
from ..sparse_structure_flow import TimestepEmbedder
from .sparse_unet_vae import (
    SparseConvNeXtBlock3d,
    SparseResBlock3d,
    SparseResBlockDownsample3d,
    SparseResBlockS2C3d,
    SparseResBlockUpsample3d,
    SparseResBlockC2S3d,
)


class SparseSkipConcatFuse(nn.Module):
    def __init__(self, channels: int, skip_channels: int):
        super().__init__()
        self.norm = LayerNorm32(channels + skip_channels, elementwise_affine=True, eps=1e-6)
        self.proj = sp.SparseLinear(channels + skip_channels, channels)

    def forward(self, x: sp.SparseTensor, skip: sp.SparseTensor) -> sp.SparseTensor:
        if x.coords.shape != skip.coords.shape or not torch.equal(x.coords, skip.coords):
            raise ValueError(
                "Skip fusion requires matching sparse coordinates: "
                f"decoder {tuple(x.coords.shape)} vs skip {tuple(skip.coords.shape)}"
            )
        h = sp.sparse_cat([x, skip], dim=-1)
        h = h.replace(self.norm(h.feats))
        return self.proj(h)


class SparseDiTBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        pe_mode: Literal["ape", "rope"] = "rope",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = True,
        initialization: str = "scaled",
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.dtype = str_to_dtype(dtype)

        self.to_latent = sp.SparseLinear(in_channels, latent_channels)
        self.input_layer = sp.SparseLinear(latent_channels, model_channels)
        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode="full",
                use_checkpoint=use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
            )
            for _ in range(num_blocks)
        ])
        self.out_layer = zero_module(sp.SparseLinear(model_channels, latent_channels))
        self.from_latent = sp.SparseLinear(latent_channels, in_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == "vanilla":
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        elif self.initialization == "scaled":
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor) -> sp.SparseTensor:
        z = self.to_latent(x)
        h = self.input_layer(z)
        h = manual_cast(h, self.dtype)

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb)

        h = manual_cast(h, z.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        z = z + self.out_layer(h)
        return self.from_latent(z)


class SparseUNetDiTFlowModel(nn.Module):
    """
    Sparse UNet-shaped flow denoiser with a timestep-conditioned DiT bottleneck.

    The sparse support is inherited from the input x_t. Downsample/upsample caches
    are produced by the encoder and reused by the decoder, so the final prediction
    is emitted on the same ground-truth sparse coordinates as x_t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: List[int],
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        latent_channels: Optional[int] = None,
        dit_model_channels: Optional[int] = None,
        dit_num_blocks: int = 12,
        dit_num_heads: Optional[int] = None,
        dit_num_head_channels: Optional[int] = 64,
        dit_mlp_ratio: float = 4.0,
        pe_mode: Literal["ape", "rope"] = "rope",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        dit_use_checkpoint: bool = False,
        dit_share_mod: bool = True,
        initialization: str = "scaled",
        qk_rms_norm: bool = False,
        use_fp16: bool = False,
    ):
        super().__init__()
        if not (
            len(model_channels)
            == len(num_blocks)
            == len(block_type)
            == len(block_args)
        ):
            raise ValueError("model_channels, num_blocks, block_type, and block_args must have the same length.")
        if len(down_block_type) != len(model_channels) - 1:
            raise ValueError("down_block_type must have one entry per encoder downsample.")
        if len(up_block_type) != len(model_channels) - 1:
            raise ValueError("up_block_type must have one entry per decoder upsample.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])

        self.encoder_blocks = nn.ModuleList([])
        for i in range(len(model_channels)):
            level = nn.ModuleList([])
            for _ in range(num_blocks[i]):
                level.append(globals()[block_type[i]](model_channels[i], **block_args[i]))
            if i < len(model_channels) - 1:
                level.append(globals()[down_block_type[i]](
                    model_channels[i],
                    model_channels[i + 1],
                    **block_args[i],
                ))
            self.encoder_blocks.append(level)

        bottleneck_channels = model_channels[-1]
        latent_channels = latent_channels or bottleneck_channels
        dit_model_channels = dit_model_channels or bottleneck_channels
        self.bottleneck = SparseDiTBottleneck(
            in_channels=bottleneck_channels,
            latent_channels=latent_channels,
            model_channels=dit_model_channels,
            num_blocks=dit_num_blocks,
            num_heads=dit_num_heads,
            num_head_channels=dit_num_head_channels,
            mlp_ratio=dit_mlp_ratio,
            pe_mode=pe_mode,
            rope_freq=rope_freq,
            dtype=dtype,
            use_checkpoint=dit_use_checkpoint,
            share_mod=dit_share_mod,
            initialization=initialization,
            qk_rms_norm=qk_rms_norm,
        )

        decoder_channels = list(reversed(model_channels))
        decoder_blocks = list(reversed(num_blocks))
        decoder_block_type = list(reversed(block_type))
        decoder_block_args = list(reversed(block_args))
        self.decoder_blocks = nn.ModuleList([])
        self.skip_fuse = nn.ModuleList([])
        for i in range(len(decoder_channels)):
            level = nn.ModuleList([])
            for _ in range(decoder_blocks[i]):
                level.append(globals()[decoder_block_type[i]](
                    decoder_channels[i],
                    **decoder_block_args[i],
                ))
            if i < len(decoder_channels) - 1:
                level.append(globals()[up_block_type[i]](
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    pred_subdiv=False,
                    **decoder_block_args[i],
                ))
                self.skip_fuse.append(SparseSkipConcatFuse(
                    decoder_channels[i + 1],
                    decoder_channels[i + 1],
                ))
            self.decoder_blocks.append(level)

        self.output_layer = zero_module(sp.SparseLinear(model_channels[0], out_channels))

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.encoder_blocks.apply(convert_module_to_f16)
        self.decoder_blocks.apply(convert_module_to_f16)
        self.skip_fuse.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.encoder_blocks.apply(convert_module_to_f32)
        self.decoder_blocks.apply(convert_module_to_f32)
        self.skip_fuse.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.input_layer.apply(_basic_init)
        self.encoder_blocks.apply(_basic_init)
        self.decoder_blocks.apply(_basic_init)
        self.skip_fuse.apply(_basic_init)
        self.output_layer.apply(_basic_init)
        nn.init.constant_(self.output_layer.weight, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def _disable_autocast(self, x: sp.SparseTensor):
        if x.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _encode(self, x: sp.SparseTensor) -> Tuple[sp.SparseTensor, List[sp.SparseTensor]]:
        h = self.input_layer(x)
        h = h.type(self.dtype)
        skips = []
        for i, level in enumerate(self.encoder_blocks):
            for j, block in enumerate(level):
                is_downsample = i < len(self.encoder_blocks) - 1 and j == len(level) - 1
                if is_downsample:
                    skips.append(h)
                h = block(h)
        return h, skips

    def _decode(self, h: sp.SparseTensor, skips: List[sp.SparseTensor], out_dtype: torch.dtype) -> sp.SparseTensor:
        h = h.type(self.dtype)
        skip_idx = len(skips) - 1
        fuse_idx = 0
        for i, level in enumerate(self.decoder_blocks):
            for j, block in enumerate(level):
                is_upsample = i < len(self.decoder_blocks) - 1 and j == len(level) - 1
                h = block(h)
                if is_upsample:
                    h = self.skip_fuse[fuse_idx](h, skips[skip_idx])
                    skip_idx -= 1
                    fuse_idx += 1
        h = h.type(out_dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.output_layer(h)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        del cond, kwargs
        out_dtype = x.dtype
        with self._disable_autocast(x):
            x = x.type(self.dtype)
            h, skips = self._encode(x)
        h = self.bottleneck(h, t)
        with self._disable_autocast(h):
            y = self._decode(h, skips, out_dtype)
        if y.coords.shape != x.coords.shape or not torch.equal(y.coords, x.coords):
            raise ValueError(
                "SparseUNetDiTFlowModel must preserve input sparse coordinates: "
                f"output {tuple(y.coords.shape)} vs input {tuple(x.coords.shape)}"
            )
        return y
