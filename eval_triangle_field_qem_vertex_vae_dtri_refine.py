"""Register the isolated d_tri refinement decoder, then run standard vertex eval."""

from trellis2 import models
from trellis2.models.sc_vaes.sparse_unet_vae_hierarchical_vertex_edge_dtri_refine import (
    SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder,
)

setattr(
    models,
    "SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder",
    SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder,
)

from eval_triangle_field_qem_vertex_vae import main


if __name__ == "__main__":
    main()

