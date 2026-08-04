import importlib

__attributes = {
    'FlexiDualGridDataset': 'flexi_dual_grid',
    'SparseVoxelPbrDataset':'sparse_voxel_pbr',
    'SparseVoxelShapeDataset':'sparse_voxel_pbr',
    'DenseGaussianPatchDataset':'sparse_voxel_pbr',
    'SparseGaussianPatchDataset':'sparse_voxel_pbr',
    'SparseVoxelOccupancyDataset': 'sparse_voxel_occupancy',
    'SparseVoxelTriangleFieldDataset': 'sparse_voxel_triangle_field',
    'MultiResolutionSparseVoxelTriangleFieldDataset': 'sparse_voxel_triangle_field',
    'TriangleFieldSuperResolutionDataset': 'triangle_field_super_resolution',
    'TriangleFieldLatentSuperResolutionDataset': 'triangle_field_super_resolution',
    'MultiResolutionTriangleFieldLatentSuperResolutionDataset': 'triangle_field_super_resolution',
    'PointDensityMeshDataset': 'point_density_mesh',
    
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    
    'SLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',
    'SLatShape': 'structured_latent_shape',
    'ImageConditionedSLatShape': 'structured_latent_shape',
    'SLatPbr': 'structured_latent_svpbr',
    'ImageConditionedSLatPbr': 'structured_latent_svpbr',
    'MichelangeloConditionedGaussianDistanceSLat': 'structured_latent_gaussian_distance',
    'MichelangeloShapeConditionedGaussianDistanceSLat': 'structured_latent_gaussian_distance',
    'MichelangeloShapeConditionedTriangleFieldSLat': 'structured_latent_triangle_field',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':    
    from .flexi_dual_grid import FlexiDualGridDataset
    from .sparse_voxel_pbr import SparseVoxelPbrDataset, SparseVoxelShapeDataset, DenseGaussianPatchDataset, SparseGaussianPatchDataset
    from .sparse_voxel_occupancy import SparseVoxelOccupancyDataset
    from .sparse_voxel_triangle_field import SparseVoxelTriangleFieldDataset, MultiResolutionSparseVoxelTriangleFieldDataset
    from .triangle_field_super_resolution import TriangleFieldSuperResolutionDataset, TriangleFieldLatentSuperResolutionDataset, MultiResolutionTriangleFieldLatentSuperResolutionDataset
    from .point_density_mesh import PointDensityMeshDataset
    
    from .sparse_structure_latent import SparseStructureLatent, ImageConditionedSparseStructureLatent
    from .structured_latent import SLat, ImageConditionedSLat
    from .structured_latent_shape import SLatShape, ImageConditionedSLatShape
    from .structured_latent_svpbr import SLatPbr, ImageConditionedSLatPbr
    from .structured_latent_gaussian_distance import MichelangeloConditionedGaussianDistanceSLat, MichelangeloShapeConditionedGaussianDistanceSLat
    from .structured_latent_triangle_field import MichelangeloShapeConditionedTriangleFieldSLat
    
