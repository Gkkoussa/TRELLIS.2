"""Rasterized visualizations for sparse vertex-occupancy predictions."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .vertex_subdivision import child_offsets


GT_COLOR = (0.10, 1.00, 0.20)
PRED_COLOR = (1.00, 0.35, 0.05)
TP_COLOR = (0.10, 1.00, 0.20)
FP_COLOR = (1.00, 0.05, 0.05)
FN_COLOR = (0.05, 0.35, 1.00)
SUPPORT_COLOR = (0.20, 0.20, 0.20)


@torch.no_grad()
def add_image_label(image: torch.Tensor, label: str) -> torch.Tensor:
    """Overlay a compact label while preserving the tensor's shape and device."""
    device = image.device
    array = (
        image.detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
        .transpose(1, 2, 0)
    )
    pil_image = Image.fromarray(array, mode='RGB')
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle((0, 0, pil_image.width, 26), fill=(255, 255, 255))
    draw.text((8, 7), label, fill=(0, 0, 0))
    labeled = np.asarray(pil_image, dtype=np.float32).copy()
    return torch.from_numpy(labeled).permute(2, 0, 1).to(device=device) / 255.0


class VertexVoxelRenderer:
    """Render sparse voxel coordinates with the repository VoxelRenderer."""

    def __init__(self, image_resolution: int = 512, ssaa: int = 4):
        if image_resolution <= 0 or ssaa <= 0:
            raise ValueError('image_resolution and ssaa must be positive.')
        # Keep the CUDA-backed renderer imports lazy.  This module's tensor
        # helpers can then be imported by normal/non-rendering runs without
        # initializing flex_gemm or a CUDA context.
        from ..renderers import VoxelRenderer
        from ..representations import Voxel
        from .render_utils import snapshot_orbit_cameras

        self.image_resolution = int(image_resolution)
        self.renderer = VoxelRenderer()
        self.voxel_type = Voxel
        self.renderer.rendering_options.resolution = self.image_resolution
        self.renderer.rendering_options.ssaa = int(ssaa)
        self.extrinsics, self.intrinsics = snapshot_orbit_cameras()

    @torch.no_grad()
    def render(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor,
        resolution: int,
        minimum_marker_pixels: int = 1,
    ) -> torch.Tensor:
        """Render exact voxel centers, optionally enlarging them in image space.

        Enlarging after rasterization preserves the original 3D voxel locations.
        Changing ``voxel_size`` on the representation would also move the voxel
        centers, because the repository's :class:`Voxel` derives positions from
        both coordinates and voxel size.
        """
        if minimum_marker_pixels <= 0:
            raise ValueError('minimum_marker_pixels must be positive.')
        device = coords.device
        sheet = torch.zeros(
            3,
            2 * self.image_resolution,
            2 * self.image_resolution,
            device=device,
            dtype=torch.float32,
        )
        if len(coords) == 0:
            return sheet
        if colors.shape != (len(coords), 3):
            raise ValueError(
                f'colors must have shape ({len(coords)}, 3), got {tuple(colors.shape)}.'
            )
        representation = self.voxel_type(
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / int(resolution),
            coords=coords.to(device=device, dtype=torch.int32).contiguous(),
            attrs=None,
            layout={'color': slice(0, 3)},
        )
        colors = colors.to(device=device, dtype=torch.float32).contiguous()
        for view_index, (extrinsic, intrinsic) in enumerate(
            zip(self.extrinsics, self.intrinsics)
        ):
            with torch.autocast(device_type='cuda', enabled=False):
                rendered = self.renderer.render(
                    representation,
                    extrinsic.float(),
                    intrinsic.float(),
                    colors_overwrite=colors,
                )['color'].float()
            row, column = divmod(view_index, 2)
            sheet[
                :,
                row * self.image_resolution:(row + 1) * self.image_resolution,
                column * self.image_resolution:(column + 1) * self.image_resolution,
            ] = rendered
        estimated_voxel_pixels = self.image_resolution / int(resolution)
        pixel_radius = max(
            0,
            int(np.ceil((int(minimum_marker_pixels) - estimated_voxel_pixels) / 2)),
        )
        # Avoid turning dense views into solid blocks when a very large target
        # size is accidentally requested.
        pixel_radius = min(pixel_radius, 4)
        if pixel_radius > 0:
            kernel_size = 2 * pixel_radius + 1
            enlarged = torch.empty_like(sheet)
            # Pool each camera tile separately so colors never bleed between
            # adjacent views in the 2x2 sheet.
            for row in range(2):
                for column in range(2):
                    y0 = row * self.image_resolution
                    x0 = column * self.image_resolution
                    tile = sheet[
                        :,
                        y0:y0 + self.image_resolution,
                        x0:x0 + self.image_resolution,
                    ]
                    enlarged[
                        :,
                        y0:y0 + self.image_resolution,
                        x0:x0 + self.image_resolution,
                    ] = F.max_pool2d(
                        tile.unsqueeze(0),
                        kernel_size=kernel_size,
                        stride=1,
                        padding=pixel_radius,
                    ).squeeze(0)
            sheet = enlarged
        return sheet


def _constant_colors(
    count: int,
    color,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(color, device=device, dtype=torch.float32).expand(count, -1)


@torch.no_grad()
def build_vertex_renderings(
    renderer: VertexVoxelRenderer,
    support_xyz: torch.Tensor,
    gt: torch.Tensor,
    prediction: torch.Tensor,
    resolution: int,
    confidence: Optional[torch.Tensor] = None,
    minimum_marker_pixels: int = 6,
    include_mistakes: bool = False,
) -> Dict[str, torch.Tensor]:
    """Render GT, prediction, errors, support context, and optional confidence."""
    support_xyz = support_xyz.long()
    gt = gt.reshape(-1).bool()
    prediction = prediction.reshape(-1).bool()
    if len(support_xyz) != len(gt) or gt.shape != prediction.shape:
        raise ValueError('Support, GT, and prediction must be aligned.')
    device = support_xyz.device

    gt_xyz = support_xyz[gt]
    pred_xyz = support_xyz[prediction]
    tp = gt & prediction
    fp = ~gt & prediction
    fn = gt & ~prediction
    error_mask = tp | fp | fn
    error_colors = torch.zeros(len(support_xyz), 3, device=device)
    error_colors[tp] = torch.tensor(TP_COLOR, device=device)
    error_colors[fp] = torch.tensor(FP_COLOR, device=device)
    error_colors[fn] = torch.tensor(FN_COLOR, device=device)

    mistake_mask = fp | fn
    context_colors = _constant_colors(len(support_xyz), SUPPORT_COLOR, device)

    error_overlay = renderer.render(
        support_xyz[error_mask],
        error_colors[error_mask],
        resolution,
        minimum_marker_pixels=minimum_marker_pixels,
    )
    support_context = renderer.render(
        support_xyz,
        context_colors,
        resolution,
    )
    # The support stays at its true rasterized size while colored vertex
    # markers are enlarged and placed over it. This avoids hiding errors in a
    # dense cloud of equally enlarged triangle-support voxels.
    overlay_mask = error_overlay.amax(dim=0, keepdim=True) > 1e-4
    support_context = torch.where(overlay_mask, error_overlay, support_context)

    images = {
        'gt': renderer.render(
            gt_xyz,
            _constant_colors(len(gt_xyz), GT_COLOR, device),
            resolution,
            minimum_marker_pixels=minimum_marker_pixels,
        ),
        'prediction': renderer.render(
            pred_xyz,
            _constant_colors(len(pred_xyz), PRED_COLOR, device),
            resolution,
            minimum_marker_pixels=minimum_marker_pixels,
        ),
        'error': error_overlay,
        'support_context': support_context,
    }
    if include_mistakes:
        images['mistakes'] = renderer.render(
            support_xyz[mistake_mask],
            error_colors[mistake_mask],
            resolution,
            minimum_marker_pixels=minimum_marker_pixels,
        )
    if confidence is not None:
        confidence = confidence.reshape(-1).float().clamp(0, 1)
        if len(confidence) != len(support_xyz):
            raise ValueError('Confidence must be aligned with support.')
        confidence_colors = torch.stack(
            [
                confidence,
                0.25 + 0.75 * confidence,
                1.0 - confidence,
            ],
            dim=1,
        ).clamp(0, 1)
        images['confidence'] = renderer.render(
            support_xyz,
            confidence_colors,
            resolution,
        )
    return images


@torch.no_grad()
def build_child_mask_renderings(
    renderer: VertexVoxelRenderer,
    gt: torch.Tensor,
    prediction: torch.Tensor,
    confidence: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Render one parent's 2x2x2 child mask in a normalized local cube."""
    offsets = child_offsets(confidence.device)
    return build_vertex_renderings(
        renderer,
        offsets,
        gt,
        prediction,
        resolution=2,
        confidence=confidence,
    )
