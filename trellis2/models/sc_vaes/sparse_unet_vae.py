from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...modules.utils import convert_module_to, convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.transformer import SparseTransformerBlock, ModulatedSparseTransformerBlock
from ...modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock
from ..shape_token_encoder import HunyuanShapeTokenEncoder


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=t.device)) *
        torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        return h + x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        use_bf16: bool = False,
        output_transformer_block: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if use_fp16 and use_bf16:
            raise ValueError('use_fp16 and use_bf16 are mutually exclusive')
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.bfloat16 if use_bf16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )

        self.output_transformer_modulated = False
        if output_transformer_block is not None:
            transformer_args = dict(output_transformer_block)
            transformer_type = transformer_args.pop('block_type', 'SparseTransformerBlock')
            transformer_depth = int(transformer_args.pop('num_blocks', 1))
            if transformer_depth <= 0:
                raise ValueError(f'output transformer num_blocks must be positive, got {transformer_depth}')
            transformer_classes = {
                'SparseTransformerBlock': SparseTransformerBlock,
                'ModulatedSparseTransformerBlock': ModulatedSparseTransformerBlock,
            }
            if transformer_type not in transformer_classes:
                raise ValueError(
                    f'Unsupported output transformer block_type {transformer_type}; '
                    f'expected one of {tuple(transformer_classes)}'
                )
            transformer_cls = transformer_classes[transformer_type]
            self.output_transformer_modulated = transformer_cls is ModulatedSparseTransformerBlock
            output_blocks = [
                transformer_cls(model_channels[-1], **transformer_args)
                for _ in range(transformer_depth)
            ]
            # Preserve legacy state-dict keys for existing one-block configs.
            self.output_transformer_block = (
                output_blocks[0] if transformer_depth == 1 else nn.ModuleList(output_blocks)
            )
        else:
            self.output_transformer_block = None

        self.initialize_weights()
        if self.output_transformer_block is not None:
            output_blocks = (
                self.output_transformer_block
                if isinstance(self.output_transformer_block, nn.ModuleList)
                else [self.output_transformer_block]
            )
            for block in output_blocks:
                if self.output_transformer_modulated:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                else:
                    zero_module(block.attn.to_out)
                    zero_module(block.mlp.mlp[2])
        if use_fp16:
            self.convert_to_fp16()
        elif use_bf16:
            self.convert_to_bf16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        if self.output_transformer_block is not None:
            self.output_transformer_block.apply(convert_module_to_f16)
        self.dtype = torch.float16

    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(lambda module: convert_module_to(module, torch.bfloat16))
        if self.output_transformer_block is not None:
            self.output_transformer_block.apply(
                lambda module: convert_module_to(module, torch.bfloat16)
            )
        self.dtype = torch.bfloat16

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        if self.output_transformer_block is not None:
            self.output_transformer_block.apply(convert_module_to_f32)
        self.dtype = torch.float32

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def _apply_output_transformer(
        self,
        h: sp.SparseTensor,
        modulation: Optional[torch.Tensor] = None,
    ) -> sp.SparseTensor:
        if self.output_transformer_block is None:
            return h
        if self.output_transformer_modulated and modulation is None:
            raise ValueError('Modulated output transformer requires timestep modulation')

        output_blocks = (
            self.output_transformer_block
            if isinstance(self.output_transformer_block, nn.ModuleList)
            else [self.output_transformer_block]
        )
        for block in output_blocks:
            h = block(h, modulation) if self.output_transformer_modulated else block(h)
        return h

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        h = self._apply_output_transformer(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z


class SparseUnetVaeEncoderTimeFiLM(SparseUnetVaeEncoder):
    """
    Sparse VAE encoder with zero-initialized timestep FiLM after each encoder block.

    The posterior interface is intentionally kept identical to SparseUnetVaeEncoder:
    to_latent outputs mean/logvar, and callers can choose mean-only or posterior
    sampling through sample_posterior.
    """
    def __init__(
        self,
        *args,
        time_embed_dim: int = 256,
        time_embed_max_period: int = 10000,
        resolution_conditioning: bool = False,
        resolution_reference: float = 128.0,
        resolution_embed_zero_init: bool = True,
        conditioning_noise_conditioning: bool = False,
        conditioning_noise_embed_scale: float = 1000.0,
        density_statistics_conditioning: bool = False,
        shape_conditioning: Optional[Dict[str, Any]] = None,
        latent_cond_channels: int = 0,
        latent_cond_mode: Optional[Literal['bottleneck']] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_embed_dim = time_embed_dim
        self.time_embed_max_period = time_embed_max_period
        self.resolution_conditioning = bool(resolution_conditioning)
        self.resolution_reference = float(resolution_reference)
        self.resolution_embed_zero_init = bool(resolution_embed_zero_init)
        self.conditioning_noise_conditioning = bool(conditioning_noise_conditioning)
        self.conditioning_noise_embed_scale = float(conditioning_noise_embed_scale)
        self.density_statistics_conditioning = bool(density_statistics_conditioning)
        self.shape_conditioning = shape_conditioning is not None
        if self.resolution_reference <= 0:
            raise ValueError(f'resolution_reference must be positive, got {resolution_reference}')
        if self.conditioning_noise_embed_scale <= 0:
            raise ValueError(
                'conditioning_noise_embed_scale must be positive, got '
                f'{conditioning_noise_embed_scale}'
            )
        self.latent_cond_channels = int(latent_cond_channels)
        self.latent_cond_mode = latent_cond_mode
        if self.latent_cond_mode not in (None, 'bottleneck'):
            raise ValueError(f"latent_cond_mode must be None or 'bottleneck', got {self.latent_cond_mode}")
        if self.latent_cond_mode is not None and self.latent_cond_channels <= 0:
            raise ValueError(f'latent_cond_channels must be positive when latent_cond_mode is set, got {latent_cond_channels}')

        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.resolution_conditioning:
            self.resolution_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            if self.resolution_embed_zero_init:
                nn.init.constant_(self.resolution_embed[-1].weight, 0)
                nn.init.constant_(self.resolution_embed[-1].bias, 0)
        if self.conditioning_noise_conditioning:
            self.conditioning_noise_embed = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        if self.density_statistics_conditioning:
            self.density_statistics_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(time_embed_dim + 2, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
                for _ in range(3)
            ])
        if self.shape_conditioning:
            shape_args = dict(shape_conditioning)
            context_points = int(shape_args.pop('context_points', 16384))
            num_tokens = int(shape_args.pop('num_tokens', 1024))
            shape_width = int(shape_args.pop('width', self.model_channels[-1]))
            shape_heads = int(shape_args.pop('heads', 16))
            shape_layers = int(shape_args.pop('encoder_layers', 8))
            num_freqs = int(shape_args.pop('num_freqs', 8))
            include_pi = bool(shape_args.pop('include_pi', True))
            qkv_bias = bool(shape_args.pop('qkv_bias', True))
            qk_norm = bool(shape_args.pop('qk_norm', False))
            use_ln_post = bool(shape_args.pop('use_ln_post', True))
            post_blocks = int(shape_args.pop('post_blocks', 2))
            cross_block_args = dict(shape_args.pop('cross_block_args', {}))
            post_block_args = dict(shape_args.pop('post_block_args', {}))
            if shape_args:
                raise ValueError(
                    f'Unknown shape_conditioning options: {sorted(shape_args)}'
                )
            if post_blocks < 0:
                raise ValueError(f'post_blocks must be non-negative, got {post_blocks}')

            bottleneck_channels = self.model_channels[-1]
            self.shape_context_points = context_points
            self.shape_num_tokens = num_tokens
            self.shape_encoder_downsample_factor = 2 ** (len(self.model_channels) - 1)
            self.shape_token_encoder = HunyuanShapeTokenEncoder(
                context_points=context_points,
                num_tokens=num_tokens,
                width=shape_width,
                heads=shape_heads,
                layers=shape_layers,
                num_freqs=num_freqs,
                include_pi=include_pi,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                use_ln_post=use_ln_post,
            )
            self.shape_query_position_proj = nn.Sequential(
                nn.Linear(self.shape_token_encoder.fourier_embedder.out_dim, bottleneck_channels),
                nn.SiLU(),
                nn.Linear(bottleneck_channels, bottleneck_channels),
            )
            cross_defaults = {
                'num_heads': 8,
                'mlp_ratio': 5.3334,
                'attn_mode': 'full',
                'use_checkpoint': False,
                'use_rope': True,
                'rope_freq': (1.0, 10000.0),
                'qk_rms_norm': True,
                'qk_rms_norm_cross': True,
            }
            cross_defaults.update(cross_block_args)
            self.shape_cross_block = ModulatedSparseTransformerCrossBlock(
                bottleneck_channels,
                ctx_channels=shape_width,
                **cross_defaults,
            )
            post_defaults = {
                'num_heads': 8,
                'mlp_ratio': 5.3334,
                'attn_mode': 'full',
                'use_checkpoint': False,
                'use_rope': True,
                'rope_freq': (1.0, 10000.0),
                'qk_rms_norm': True,
            }
            post_defaults.update(post_block_args)
            self.shape_post_blocks = nn.ModuleList([
                ModulatedSparseTransformerBlock(
                    bottleneck_channels,
                    **post_defaults,
                )
                for _ in range(post_blocks)
            ])
            self.shape_condition_time_proj = nn.Linear(
                self.time_embed_dim,
                bottleneck_channels,
            )
            nn.init.normal_(self.shape_condition_time_proj.weight, std=0.02)
            nn.init.constant_(self.shape_condition_time_proj.bias, 0)

            # Preserve the pretrained encoder function at initialization. The
            # zero output projection receives gradients immediately; earlier
            # shape modules begin receiving gradients after its first update.
            nn.init.constant_(self.shape_cross_block.cross_attn.to_out.weight, 0)
            nn.init.constant_(self.shape_cross_block.cross_attn.to_out.bias, 0)
            nn.init.constant_(self.shape_cross_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.shape_cross_block.adaLN_modulation[-1].bias, 0)
            for block in self.shape_post_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            self._convert_shape_modules(self.dtype)
        self.film_layers = nn.ModuleList([])
        for i, res in enumerate(self.blocks):
            film_res = nn.ModuleList([])
            for j, _ in enumerate(res):
                out_channels = self.model_channels[i]
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    out_channels = self.model_channels[i + 1]
                film = nn.Linear(time_embed_dim, 2 * out_channels)
                nn.init.constant_(film.weight, 0)
                nn.init.constant_(film.bias, 0)
                film_res.append(film)
            self.film_layers.append(film_res)
        if self.latent_cond_mode == 'bottleneck':
            self.latent_bottleneck_proj = sp.SparseLinear(
                self.model_channels[-1] + self.latent_cond_channels,
                self.model_channels[-1],
            )
            nn.init.constant_(self.latent_bottleneck_proj.weight, 0)
            nn.init.constant_(self.latent_bottleneck_proj.bias, 0)
            with torch.no_grad():
                self.latent_bottleneck_proj.weight[:, :self.model_channels[-1]].copy_(
                    torch.eye(self.model_channels[-1])
                )
        if self.output_transformer_modulated:
            self.output_transformer_time_proj = nn.Linear(
                self.time_embed_dim,
                self.model_channels[-1],
            )
            nn.init.normal_(self.output_transformer_time_proj.weight, std=0.02)
            nn.init.constant_(self.output_transformer_time_proj.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the sparse torso to float16 while leaving time MLPs in fp32.
        """
        self.blocks.apply(convert_module_to_f16)
        if self.output_transformer_block is not None:
            self.output_transformer_block.apply(convert_module_to_f16)
        self._convert_shape_modules(torch.float16)
        self.dtype = torch.float16

    def convert_to_bf16(self) -> None:
        super().convert_to_bf16()
        self._convert_shape_modules(torch.bfloat16)

    def convert_to_fp32(self) -> None:
        super().convert_to_fp32()
        self._convert_shape_modules(torch.float32)

    def _convert_shape_modules(self, dtype: torch.dtype) -> None:
        if not hasattr(self, 'shape_token_encoder'):
            return
        converter = lambda module: convert_module_to(module, dtype)
        # Hunyuan uses ordinary nn.LayerNorm rather than LayerNorm32, so its
        # normalization parameters must match the projected token dtype.
        self.shape_token_encoder.to(dtype=dtype)
        self.shape_query_position_proj.apply(converter)
        self.shape_cross_block.apply(converter)
        self.shape_post_blocks.apply(converter)
        # Match the existing FiLM projections: consume the fp32 timestep
        # embedding, then cast the projected modulation at the call site.

    def _apply_film(self, h: sp.SparseTensor, emb: torch.Tensor, i: int, j: int) -> sp.SparseTensor:
        scale_shift = self.film_layers[i][j](emb).to(dtype=h.feats.dtype)
        scale, shift = scale_shift.chunk(2, dim=-1)
        batch_idx = h.coords[:, 0].long()
        feats = h.feats * (1.0 + scale[batch_idx]) + shift[batch_idx]
        return h.replace(feats)

    def _embed_resolution(
        self,
        resolution: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.resolution_conditioning:
            return None
        if resolution is None:
            raise ValueError('resolution must be provided when resolution_conditioning is enabled')
        resolution = torch.as_tensor(resolution, device=device, dtype=torch.float32).reshape(-1)
        if resolution.numel() == 1 and batch_size > 1:
            resolution = resolution.expand(batch_size)
        if resolution.numel() != batch_size:
            raise ValueError(
                f'resolution must have one value per sample, got {resolution.numel()} for batch {batch_size}'
            )
        if torch.any(resolution <= 0):
            raise ValueError('resolution values must be positive')
        resolution_scalar = torch.log2(resolution / self.resolution_reference).unsqueeze(-1)
        return self.resolution_embed(resolution_scalar)

    def _embed_conditioning_noise(
        self,
        conditioning_noise_level: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.conditioning_noise_conditioning:
            return None
        if conditioning_noise_level is None:
            raise ValueError(
                'conditioning_noise_level must be provided when '
                'conditioning_noise_conditioning is enabled'
            )
        level = torch.as_tensor(
            conditioning_noise_level,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
        if level.numel() == 1 and batch_size > 1:
            level = level.expand(batch_size)
        if level.numel() != batch_size:
            raise ValueError(
                'conditioning_noise_level must have one value per sample, got '
                f'{level.numel()} for batch {batch_size}'
            )
        if torch.any((level < 0) | (level > 1)):
            raise ValueError('conditioning_noise_level values must be in [0, 1]')
        noise_emb = timestep_embedding(
            level * self.conditioning_noise_embed_scale,
            self.time_embed_dim,
            self.time_embed_max_period,
        )
        return self.conditioning_noise_embed(noise_emb)

    def _embed_density_statistics(
        self,
        density_statistics: Optional[torch.Tensor],
        density_statistics_presence: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.density_statistics_conditioning:
            return None
        if density_statistics is None:
            raise ValueError(
                'density_statistics must be provided when '
                'density_statistics_conditioning is enabled'
            )
        values = torch.as_tensor(
            density_statistics,
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size, -1)
        if values.shape != (batch_size, 3):
            raise ValueError(
                f'density_statistics must have shape ({batch_size}, 3), got {tuple(values.shape)}'
            )
        if density_statistics_presence is None:
            presence = torch.ones_like(values)
        else:
            presence = torch.as_tensor(
                density_statistics_presence,
                device=device,
                dtype=torch.float32,
            ).reshape(batch_size, -1)
            if presence.shape != values.shape:
                raise ValueError(
                    'density_statistics_presence must match density_statistics shape, got '
                    f'{tuple(presence.shape)} and {tuple(values.shape)}'
                )
            if torch.any((presence < 0) | (presence > 1)):
                raise ValueError('density_statistics_presence values must be in [0, 1]')
        if not torch.isfinite(values).all():
            raise ValueError('density_statistics contains non-finite values')

        result = torch.zeros(
            (batch_size, self.time_embed_dim),
            device=device,
            dtype=torch.float32,
        )
        for index, embed in enumerate(self.density_statistics_embed):
            value = values[:, index]
            available = presence[:, index:index + 1]
            fourier = timestep_embedding(
                value,
                self.time_embed_dim,
                self.time_embed_max_period,
            )
            embed_input = torch.cat([
                fourier * available,
                value.unsqueeze(-1) * available,
                available,
            ], dim=-1)
            result = result + embed(embed_input)
        return result

    @staticmethod
    def _check_matching_coords(a: sp.SparseTensor, b: sp.SparseTensor, name_a: str, name_b: str) -> None:
        if not torch.equal(a.coords, b.coords):
            raise ValueError(
                f'{name_a} and {name_b} coords must match, got {a.coords.shape} vs {b.coords.shape}'
            )

    @staticmethod
    def normalized_bottleneck_centers(
        coords: torch.Tensor,
        resolution: torch.Tensor,
        downsample_factor: int,
    ) -> torch.Tensor:
        """Map bottleneck voxel coordinates into normalized object space."""
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f'coords must have shape [N, 4], got {tuple(coords.shape)}')
        if downsample_factor <= 0:
            raise ValueError(f'downsample_factor must be positive, got {downsample_factor}')
        batch_size = int(coords[:, 0].max().item()) + 1 if coords.numel() else 0
        resolution = torch.as_tensor(
            resolution,
            device=coords.device,
            dtype=torch.float32,
        ).reshape(-1)
        if resolution.numel() == 1 and batch_size > 1:
            resolution = resolution.expand(batch_size)
        if resolution.numel() != batch_size:
            raise ValueError(
                f'resolution must have one value per sample, got {resolution.numel()} '
                f'for batch {batch_size}'
            )
        if torch.any(resolution <= 0) or torch.any(
            resolution.remainder(float(downsample_factor)) != 0
        ):
            raise ValueError(
                f'resolution must be positive and divisible by {downsample_factor}'
            )
        bottleneck_resolution = resolution / float(downsample_factor)
        voxel_resolution = bottleneck_resolution[coords[:, 0].long()].unsqueeze(-1)
        centers = (coords[:, 1:].float() + 0.5) / voxel_resolution - 0.5
        tolerance = 1e-5
        if torch.any(centers < -0.5 - tolerance) or torch.any(centers > 0.5 + tolerance):
            raise ValueError('Bottleneck coordinates fall outside normalized object space')
        return centers

    def encode_shape(
        self,
        shape_points: torch.Tensor,
        shape_normals: torch.Tensor,
        *,
        random_start_point: Optional[bool] = None,
    ) -> torch.Tensor:
        if not self.shape_conditioning:
            raise ValueError('encode_shape requires shape_conditioning to be enabled')
        tokens, _ = self.shape_token_encoder(
            shape_points,
            shape_normals,
            random_start_point=random_start_point,
        )
        return tokens

    def _apply_shape_conditioning(
        self,
        h: sp.SparseTensor,
        emb: torch.Tensor,
        resolution: Optional[torch.Tensor],
        shape_points: Optional[torch.Tensor],
        shape_normals: Optional[torch.Tensor],
        shape_tokens: Optional[torch.Tensor],
        shape_presence: Optional[torch.Tensor],
    ) -> sp.SparseTensor:
        if not self.shape_conditioning:
            if any(value is not None for value in (
                shape_points,
                shape_normals,
                shape_tokens,
                shape_presence,
            )):
                raise ValueError('Shape inputs were provided but shape_conditioning is disabled')
            return h
        if resolution is None:
            raise ValueError('Shape conditioning requires the high-resolution value')
        if shape_tokens is not None and (shape_points is not None or shape_normals is not None):
            raise ValueError('Provide either cached shape_tokens or raw shape points, not both')
        if shape_tokens is None:
            if shape_points is None or shape_normals is None:
                raise ValueError(
                    'Shape conditioning requires shape_tokens or both shape_points and shape_normals'
                )
            shape_tokens = self.encode_shape(shape_points, shape_normals)
        if shape_tokens.ndim != 3:
            raise ValueError(
                f'shape_tokens must have shape [B, L, C], got {tuple(shape_tokens.shape)}'
            )
        batch_size = h.shape[0]
        if shape_tokens.shape[:2] != (batch_size, self.shape_num_tokens):
            raise ValueError(
                f'Expected shape tokens [{batch_size}, {self.shape_num_tokens}, C], '
                f'got {tuple(shape_tokens.shape)}'
            )
        if shape_presence is None:
            shape_presence = torch.ones(
                batch_size,
                device=h.feats.device,
                dtype=torch.float32,
            )
        else:
            shape_presence = torch.as_tensor(
                shape_presence,
                device=h.feats.device,
                dtype=torch.float32,
            ).reshape(-1)
            if shape_presence.numel() != batch_size:
                raise ValueError(
                    f'shape_presence must have {batch_size} values, got '
                    f'{shape_presence.numel()}'
                )
            if torch.any((shape_presence < 0) | (shape_presence > 1)):
                raise ValueError('shape_presence values must be in [0, 1]')
        context = shape_tokens.to(device=h.feats.device, dtype=h.feats.dtype)
        context = context * shape_presence.to(context.dtype)[:, None, None]

        centers = self.normalized_bottleneck_centers(
            h.coords,
            resolution,
            self.shape_encoder_downsample_factor,
        )
        position_features = self.shape_token_encoder.fourier_embedder(centers)
        position_features = self.shape_query_position_proj(
            position_features.to(dtype=h.feats.dtype)
        )
        positioned_h = h.replace(h.feats + position_features)
        modulation = self.shape_condition_time_proj(emb).to(dtype=h.feats.dtype)
        conditioned_h = self.shape_cross_block(positioned_h, modulation, context)
        # Remove the temporary absolute-position residual; only the learned
        # attention/MLP residuals are applied to the original sparse features.
        h = h.replace(h.feats + conditioned_h.feats - positioned_h.feats)
        for block in self.shape_post_blocks:
            h = block(h, modulation)
        return h

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        sample_posterior: bool = False,
        return_raw: bool = False,
        latent_cond: Optional[sp.SparseTensor] = None,
        resolution: Optional[torch.Tensor] = None,
        resolution_condition: Optional[torch.Tensor] = None,
        conditioning_noise_level: Optional[torch.Tensor] = None,
        density_statistics: Optional[torch.Tensor] = None,
        density_statistics_presence: Optional[torch.Tensor] = None,
        shape_points: Optional[torch.Tensor] = None,
        shape_normals: Optional[torch.Tensor] = None,
        shape_tokens: Optional[torch.Tensor] = None,
        shape_presence: Optional[torch.Tensor] = None,
    ):
        if t.ndim != 1:
            t = t.reshape(-1)
        emb = timestep_embedding(t, self.time_embed_dim, self.time_embed_max_period)
        emb = self.time_embed(emb)
        resolution_emb = self._embed_resolution(
            resolution if resolution_condition is None else resolution_condition,
            t.shape[0],
            t.device,
        )
        if resolution_emb is not None:
            emb = emb + resolution_emb
        conditioning_noise_emb = self._embed_conditioning_noise(
            conditioning_noise_level,
            t.shape[0],
            t.device,
        )
        if conditioning_noise_emb is not None:
            emb = emb + conditioning_noise_emb
        density_statistics_emb = self._embed_density_statistics(
            density_statistics,
            density_statistics_presence,
            t.shape[0],
            t.device,
        )
        if density_statistics_emb is not None:
            emb = emb + density_statistics_emb

        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if self.latent_cond_mode == 'bottleneck' and i == len(self.blocks) - 1:
                if latent_cond is None:
                    raise ValueError('latent_cond must be provided when latent_cond_mode="bottleneck"')
                self._check_matching_coords(h, latent_cond, 'bottleneck h', 'latent_cond')
                h = h.type(x.dtype)
                cond_feats = latent_cond.feats.to(device=h.feats.device, dtype=h.feats.dtype)
                h = h.replace(torch.cat([h.feats, cond_feats], dim=-1))
                h = self.latent_bottleneck_proj(h)
                h = h.type(self.dtype)
            for j, block in enumerate(res):
                h = block(h)
                h = self._apply_film(h, emb, i, j)
        h = self._apply_shape_conditioning(
            h,
            emb,
            resolution,
            shape_points,
            shape_normals,
            shape_tokens,
            shape_presence,
        )
        output_modulation = None
        if self.output_transformer_modulated:
            output_modulation = self.output_transformer_time_proj(emb).to(dtype=h.feats.dtype)
        h = self._apply_output_transformer(h, output_modulation)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)

        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)

        if return_raw:
            return z, mean, logvar
        else:
            return z
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.low_vram = False
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h
    
    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                else:
                    h = block(h)


class SparseUnetVaeDecoderTimeFiLM(SparseUnetVaeDecoder):
    """VAE decoder with zero-initialized timestep and resolution FiLM."""

    def __init__(
        self,
        *args,
        time_embed_dim: int = 256,
        time_embed_max_period: int = 10000,
        resolution_reference: float = 128.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_embed_dim = int(time_embed_dim)
        self.time_embed_max_period = int(time_embed_max_period)
        self.resolution_reference = float(resolution_reference)
        if self.time_embed_dim <= 0:
            raise ValueError(f'time_embed_dim must be positive, got {time_embed_dim}')
        if self.resolution_reference <= 0:
            raise ValueError(
                f'resolution_reference must be positive, got {resolution_reference}'
            )

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.resolution_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.film_layers = nn.ModuleList([])
        for i, res in enumerate(self.blocks):
            film_res = nn.ModuleList([])
            for j, _ in enumerate(res):
                out_channels = self.model_channels[i]
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    out_channels = self.model_channels[i + 1]
                film = nn.Linear(self.time_embed_dim, 2 * out_channels)
                nn.init.constant_(film.weight, 0)
                nn.init.constant_(film.bias, 0)
                film_res.append(film)
            self.film_layers.append(film_res)

    def _embed_resolution(
        self,
        resolution: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if resolution is None:
            raise ValueError('resolution must be provided to the timestep-conditioned decoder')
        resolution = torch.as_tensor(
            resolution,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
        if resolution.numel() == 1 and batch_size > 1:
            resolution = resolution.expand(batch_size)
        if resolution.numel() != batch_size:
            raise ValueError(
                f'resolution must have one value per sample, got '
                f'{resolution.numel()} for batch {batch_size}'
            )
        if torch.any(resolution <= 0):
            raise ValueError('resolution values must be positive')
        scalar = torch.log2(resolution / self.resolution_reference).unsqueeze(-1)
        return self.resolution_embed(scalar)

    def _apply_film(
        self,
        h: sp.SparseTensor,
        emb: torch.Tensor,
        i: int,
        j: int,
    ) -> sp.SparseTensor:
        scale_shift = self.film_layers[i][j](emb).to(dtype=h.feats.dtype)
        scale, shift = scale_shift.chunk(2, dim=-1)
        batch_idx = h.coords[:, 0].long()
        return h.replace(
            h.feats * (1.0 + scale[batch_idx]) + shift[batch_idx]
        )

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        resolution: torch.Tensor,
        resolution_condition: Optional[torch.Tensor] = None,
        guide_subs: Optional[List[sp.SparseTensor]] = None,
        return_subs: bool = False,
    ) -> sp.SparseTensor:
        if guide_subs is not None and self.pred_subdiv:
            raise ValueError('Only decoders with pred_subdiv=False can use guide_subs')
        if return_subs and not self.pred_subdiv:
            raise ValueError('Only decoders with pred_subdiv=True can return subdivisions')
        t = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        if t.numel() != x.shape[0]:
            raise ValueError(
                f't must have one value per sample, got {t.numel()} for batch {x.shape[0]}'
            )
        emb = self.time_embed(
            timestep_embedding(t, self.time_embed_dim, self.time_embed_max_period)
        )
        emb = emb + self._embed_resolution(
            resolution if resolution_condition is None else resolution_condition,
            x.shape[0],
            x.device,
        )

        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        guide = guide_subs[i] if guide_subs is not None else None
                        h = block(h, subdiv=guide)
                else:
                    h = block(h)
                h = self._apply_film(h, emb, i, j)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        if return_subs:
            return h, subs
        return h
       
