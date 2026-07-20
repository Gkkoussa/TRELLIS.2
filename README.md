![](assets/teaser.webp)

# Native and Compact Structured Latents for 3D Generation

<a href="https://arxiv.org/abs/2512.14692"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/microsoft/TRELLIS.2-4B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="https://huggingface.co/spaces/microsoft/TRELLIS.2"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet"></a>
<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

https://github.com/user-attachments/assets/63b43a7e-acc7-4c81-a900-6da450527d8f

*(Compressed version due to GitHub size limits. See the full-quality video on our project page!)*

**TRELLIS.2** is a state-of-the-art large 3D generative model (4B parameters) designed for high-fidelity **image-to-3D** generation. It leverages a novel "field-free" sparse voxel structure termed **O-Voxel** to reconstruct and generate arbitrary 3D assets with complex topologies, sharp features, and full PBR materials.


## ✨ Features

### 1. High Quality, Resolution & Efficiency
Our 4B-parameter model generates high-resolution fully textured assets with exceptional fidelity and efficiency using vanilla DiTs. It utilizes a Sparse 3D VAE with 16× spatial downsampling to encode assets into a compact latent space.

| Resolution | Total Time* | Breakdown (Shape + Mat) |
| :--- | :--- | :--- |
| **512³** | **~3s** | 2s + 1s |
| **1024³** | **~17s** | 10s + 7s |
| **1536³** | **~60s** | 35s + 25s |

<small>*Tested on NVIDIA H100 GPU.</small>

### 2. Arbitrary Topology Handling
The **O-Voxel** representation breaks the limits of iso-surface fields. It robustly handles complex structures without lossy conversion:
*   ✅ **Open Surfaces** (e.g., clothing, leaves)
*   ✅ **Non-manifold Geometry**
*   ✅ **Internal Enclosed Structures**

### 3. Rich Texture Modeling
Beyond basic colors, TRELLIS.2 models arbitrary surface attributes including **Base Color, Roughness, Metallic, and Opacity**, enabling photorealistic rendering and transparency support.

### 4. Minimalist Processing
Data processing is streamlined for instant conversions that are fully **rendering-free** and **optimization-free**.
*   **< 10s** (Single CPU): Textured Mesh → O-Voxel
*   **< 100ms** (CUDA): O-Voxel → Textured Mesh


## 🗺️ Roadmap

- [x] Paper release
- [x] Release image-to-3D inference code
- [x] Release pretrained checkpoints (4B)
- [x] Hugging Face Spaces demo
- [x] Release shape-conditioned texture generation inference code
- [x] Release training code


## 🛠️ Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 24GB of memory is necessary. The code has been verified on NVIDIA A100 and H100 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain packages. Recommended version is 12.4.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ```

2. Install the dependencies:
    
    **Before running the following command there are somethings to note:**
    - By adding `--new-env`, a new conda environment named `trellis2` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default the `trellis2` environment will use pytorch 2.6.0 with CUDA 12.4. If you want to use a different version of CUDA, you can remove the `--new-env` flag and manually install the required dependencies. Refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - If you have multiple CUDA Toolkit versions installed, `CUDA_HOME` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 12.4 and 13.0 installed, you can run `export CUDA_HOME=/usr/local/cuda-12.4` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs do not support `flash-attn` (e.g., NVIDIA V100), you can install `xformers` manually and set the `ATTN_BACKEND` environment variable to `xformers` before running the code. See the [Minimal Example](#minimal-example) for more details.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - If you encounter any issues during the installation, feel free to open an issue or contact us.
    
    Create a new conda environment named `trellis2` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --flash-attn            Install flash-attention
        --cumesh                Install cumesh
        --o-voxel               Install o-voxel
        --flexgemm              Install flexgemm
        --nvdiffrast            Install nvdiffrast
        --nvdiffrec             Install nvdiffrec
    ```

## 📦 Pretrained Weights

The pretrained model **TRELLIS.2-4B** is available on Hugging Face. Please refer to the model card there for more details.

| Model | Parameters | Resolution | Link |
| :--- | :--- | :--- | :--- |
| **TRELLIS.2-4B** | 4 Billion | 512³ - 1536³ | [Hugging Face](https://huggingface.co/microsoft/TRELLIS.2-4B) |

## 🧪 Triangle-Field SR Project State

This fork also contains the current ObjXL4K triangle-field super-resolution
experiments. The detailed end-to-end training notes live in
[docs/triangle_field_super_resolution_workflow.md](docs/triangle_field_super_resolution_workflow.md).
This section is the short “what should I use right now?” snapshot.

Unless stated otherwise:

* Training uses the filtered train split only.
* Evaluation uses the filtered test split, plus the no-train-duplicate filter for comparison figures.
* `$ROOT` means `/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k`.

### Best VAE: allres inv-area full-aux

The best triangle-field VAE latent space is the all-resolution inverse-area
full-aux VAE:

```bash
$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251
```

Useful checkpoint pair:

```bash
$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251/ckpts/encoder_step0320000.pt
$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251/ckpts/decoder_step0320000.pt
```

Local config:

```bash
configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_allres_objxl4k_invarea_fullaux.json
```

Important training options:

* Trained as a multi-resolution VAE over `32, 64, 128, 256, 512`.
* Fine-tuned from the 512 inverse-area aux-drop VAE:
  `$ROOT/outputs/triangle_field_vae_512_invarea_auxdrop_52039231/ckpts/{encoder,decoder}_step0180000.pt`.
* Uses the full 20-channel triangle-field input; aux dropout is disabled.
* Uses L1 reconstruction loss with inverse-triangle-area voxel weighting:
  `type=inverse_triangle_area`, `normalize=mean`, `clamp_max=100`.
* Uses 32 latent channels and `pred_subdiv=false`, so the decoder expects the saved sparse support/cache from the encoding path.
* Uses `lambda_kl=1e-4`, fp16 mixed precision, and EMA `0.9999`.

Existing 128-resolution latents from this VAE:

```bash
$ROOT/triangle_field_latents/triangle_field_vae_allres_invarea_fullaux_52454251_step0320000_128
```

### Best unconditional/remeshing SR: allres noise-only

The best non-density latent SR model is:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_noiseonly_continue_52960154
```

Useful checkpoint:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_noiseonly_continue_52960154/ckpts/encoder_ema0.9999_step0200000.pt
```

Local config:

```bash
configs/gen/triangle_field_latent_sr_flow_64to128_film_selfcond_input_allres_fullaux_noiseonly_continue_f16c32_fp16_objxl4k.json
```

Important training options:

* Uses the allres inv-area full-aux VAE decoder at step `0320000`.
* Uses latent SR flow from `64 -> 128`; higher resolutions are reached by cascaded eval.
* Uses input-style latent self-conditioning:
  `latent_self_conditioning.mode=input`, `upsample_factor=16`.
* Encoder input has 36 channels: decoded/noisy current latent features, low-res conditioning features, and self-conditioning latent features.
* Uses noise-only conditioning augmentation:
  `conditioning_augmentation.type=sparse_blur_noise`,
  `noise_level=0.25`, `disable_blur=true`, `apply_prob=1.0`.
* Uses `cond_drop_prob=0.1`, L1 loss, velocity-parameterized latent loss,
  `latent_loss_t_min=0.05`, and batched VAE cache decode.
* Continued from the earlier allres input SR run:
  `$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_52494005/ckpts/encoder_step0130000.pt`.

For remeshing-style evaluation from mesh support only, this is the main
unconditional baseline because it does not require a density field.

### Best conditional SR: allres density

The best density-conditioned latent SR model is:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_noiseonly_density128_scratch_53303600
```

Useful checkpoint:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_noiseonly_density128_scratch_53303600/ckpts/encoder_ema0.9999_step0140000.pt
```

Local config:

```bash
configs/gen/triangle_field_latent_sr_flow_64to128_film_selfcond_input_allres_fullaux_noiseonly_continue_density128_f16c32_fp16_objxl4k.json
```

Important training options:

* Same base architecture and noise-only conditioning augmentation as the allres noise-only model.
* Adds `density_conditioning=true` and reads channel `density_field` from
  `density_triangle_field_voxel`.
* Encoder input has 37 channels: the 36 allres-noise-only channels plus one density channel.
* Initialized from the allres noise-only checkpoint at step `0200000`, but trained in a separate density output folder rather than resuming an old density run.
* Uses the same latent self-conditioning, `cond_drop_prob=0.1`, velocity-parameterized latent loss, `latent_loss_t_min=0.05`, and allres VAE decoder step `0320000`.

The 128-resolution density-field payload is:

```bash
$ROOT/triangle_field_voxels_density_field/triangle_field_voxels_128
```

### Density field definition

Density fields are generated by:

```bash
voxelize_triangle_field_density_128_objxl4k.sh
```

and implemented in:

```bash
data_toolkit/voxelize_triangle_field.py
```

Run the existing 128-resolution GT-density job with:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
sbatch voxelize_triangle_field_density_128_objxl4k.sh
```

The script defaults to:

```bash
DENSITY_ROOT=$ROOT/triangle_field_voxels_density_field
RESOLUTION=128
INSTANCES=$ROOT/instances_no_triangle_dense.txt
MAX_WORKERS=8
```

It calls `data_toolkit/voxelize_triangle_field.py` with:

```text
--include_density_field
--feature_dtype float16
--npz_compression zstd
--candidate_source native
--projection_mode inside_barycentric
```

and then rebuilds metadata with `data_toolkit/build_metadata.py`. The expected
payload and metadata live under:

```bash
$ROOT/triangle_field_voxels_density_field/triangle_field_voxels_128
$ROOT/triangle_field_voxels_density_field/triangle_field_voxels_128/metadata.csv
```

For each mesh vertex `v`, compute:

```text
A_v = mean(area of triangles incident to v)
```

For each active voxel center `p`, project to the nearest triangle
`T=(v0,v1,v2)`, compute clamped/renormalized barycentric weights `b_i`, and
interpolate:

```text
area_value = b0 A_v0 + b1 A_v1 + b2 A_v2
density_field = -log(area_value / voxel_size^2 + 1e-8)
voxel_size = 1 / resolution
```

For evaluation at a higher resolution `R` from a base density value at
resolution `B=128`, the voxel-size-scaled option uses:

```text
density_R = density_128 - 2 * log(R / 128)
```

Use this when the same underlying local triangle area should have the same
meaning across resolutions. The unscaled option simply duplicates/maps density
values to the target support without changing the values.

### Evaluation tools

Use these scripts for the current SR evaluation workflows:

```bash
eval_triangle_field_latent_sr_flow.py
```

Single-stage latent SR eval, usually `64 -> 128`. Key options include
`--run_dir`, `--root`, `--split test`, `--instances`, `--ckpt`, `--ema_rate`,
`--low_resolution`, `--high_resolution`, `--latent_name`, `--steps`,
`--guidance_strength`, `--apply_conditioning_augmentation`, and
`--disable_dataset_density_conditioning`.

```bash
eval_triangle_field_latent_sr_cascade.py
```

Regular cascade eval without per-stage repeats. Use this when each stage should
run once. Key options include `--steps`, `--base_guidance_strength`,
`--guidance_strength`, `--apply_conditioning_augmentation`, `--num_samples`,
`--batch_size`, `--instances`, and `--output_dir`.

```bash
eval_triangle_field_latent_sr_stage_repeat_cascade.py
```

Repeat-cascade eval. This starts from unconditional `128`, averages to `64`,
then runs `64 -> 128`, `128 -> 256`, and `256 -> 512`, repeating each stage
with average-downsample feedback. The best current setting for comparison
figures has been:

```text
--stage_repeats 3
--steps 11
--guidance_strength 1.0
--base_guidance_strength 0.0
--apply_conditioning_augmentation
```

Density options:

* `--oracle_density_conditioning` loads density fields from a density voxel directory.
* `--constant_density_conditioning` feeds a constant density value on the target support.
* `--oracle_density_scale_mode {none,voxel_size}` and
  `--constant_density_scale_mode {none,voxel_size}` control whether
  `density_R = density_base - 2 log(R/base_R)` is applied.
* `--oracle_density_base_resolution` and `--constant_density_base_resolution`
  set the resolution where the density value is defined, usually `128`.

```bash
eval_obj_folder_latent_sr_stage_repeat_cascade.py
```

Remeshing-style support-only cascade eval from a folder of `.obj` or `.glb`
files. It extracts sparse voxel support at `128, 256, 512` with `trimesh`,
does not use GT triangle fields or GT latents, and then runs the same
repeat-cascade path. Key options include `--mesh_dir`, `--run_dir`,
`--stage_repeats`, `--max_active_voxels`, `--constant_density_conditioning`,
`--constant_density_value`, `--constant_density_scale_mode`, and
`--support_cache_dir`. If meshes are edited in-place, delete the relevant
`support_cache` directory or use a fresh `--output_dir`.

For the current density remeshing sweeps over test meshes, the typical command
uses:

```text
--stage_repeats 12
--steps 11
--guidance_strength 1
--base_guidance_strength 0
--max_active_voxels 2000000
--constant_density_conditioning
--constant_density_base_resolution 128
--constant_density_scale_mode voxel_size
--apply_conditioning_augmentation
--conditioning_augmentation_disable_blur
--conditioning_augmentation_noise_level 0.25
```


## 🚀 Usage

### 1. Image to 3D Generation

#### Minimal Example

Here is an [example](example.py) of how to use the pretrained models for 3D asset generation.

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216) # nvdiffrast limit

# 4. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 5. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)
```

Upon execution, the script generates the following files:
 - `sample.mp4`: A video visualizing the generated 3D asset with PBR materials and environmental lighting.
 - `sample.glb`: The extracted PBR-ready 3D asset in GLB format.

**Note:** The `.glb` file is exported in `OPAQUE` mode by default. Although the alpha channel is preserved within the texture map, it is not active initially. To enable transparency, import the asset into your 3D software and manually connect the texture's alpha channel to the material's opacity or alpha input.

#### Web Demo

[app.py](app.py) provides a simple web demo for image to 3D asset generation. you can run the demo with the following command:
```sh
python app.py
```

Then, you can access the demo at the address shown in the terminal.

### 2. PBR Texture Generation

Please refer to the [example_texturing.py](example_texturing.py) for an example of how to generate PBR textures for a given 3D shape. Also, you can use the [app_texturing.py](app_texturing.py) to run a web demo for PBR texture generation.


## 🏋️ Training

We provide the full training codebase, enabling users to train **TRELLIS.2** from scratch or fine-tune it on custom datasets.

### 1. Data Preparation

Before training, raw 3D assets must be converted into the **O-Voxel** representation. This process includes mesh conversion, compact structured latent generation, and metadata preparation.

> 📂 **Please refer to [data_toolkit/README.md](data_toolkit/README.md) for detailed instructions on data preprocessing and dataset organization.**

### 2. Running Training

Training is managed through the `train.py` script, which accepts multiple command-line arguments to configure experiments:

* `--config`: Path to the experiment configuration file.
* `--output_dir`: Directory for training outputs.
* `--load_dir`: Directory to load checkpoints from (defaults to `output_dir`).
* `--ckpt`: Checkpoint step to resume from (defaults to the latest).
* `--data_dir`: Dataset path or a JSON string specifying dataset locations.
* `--auto_retry`: Number of automatic retries upon failure.
* `--tryrun`: Perform a dry run without actual training.
* `--profile`: Enable training profiling.
* `--num_nodes`: Number of nodes for distributed training.
* `--node_rank`: Rank of the current node.
* `--num_gpus`: Number of GPUs per node (defaults to all available GPUs).
* `--master_addr`: Master node address for distributed training.
* `--master_port`: Port for distributed training communication.


### SC-VAE Training


To train the shape SC-VAE, run:

```sh
python train.py \
  --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json \
  --output_dir results/shape_vae_next_dc_f16c32_fp16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"mesh_dump\": \"datasets/ObjaverseXL_sketchfab/mesh_dumps\", \"dual_grid\": \"datasets/ObjaverseXL_sketchfab/dual_grid_256\", \"asset_stats\": \"datasets/ObjaverseXL_sketchfab/asset_stats\"}}"
```

This command trains the shape SC-VAE on the **Objaverse-XL** dataset using the `shape_vae_next_dc_f16c32_fp16.json` configuration. Training outputs will be saved to `results/shape_vae_next_dc_f16c32_fp16`.

The dataset is specified as a JSON string, where each dataset entry includes:

* `base`: Root directory of the dataset.
* `mesh_dump`: Directory containing preprocessed mesh dumps.
* `dual_grid`: Directory with precomputed dual-grid representations.
* `asset_stats`: Directory containing precomputed asset statistics.

To fine-tune the model at a higher resolution, use the `shape_vae_next_dc_f16c32_fp16_ft_512.json` configuration. Remember to update the `finetune_ckpt` field and adjust the dataset paths accordingly.


To train the texture SC-VAE, run:

```sh
python train.py \
  --config configs/scvae/tex_vae_next_dc_f16c32_fp16.json \
  --output_dir results/tex_vae_next_dc_f16c32_fp16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"pbr_dump\": \"datasets/ObjaverseXL_sketchfab/pbr_dumps\", \"pbr_voxel\": \"datasets/ObjaverseXL_sketchfab/pbr_voxels_256\", \"asset_stats\": \"datasets/ObjaverseXL_sketchfab/asset_stats\"}}"
```


### Flow Model Training

To train the sparse structure flow model, run:

```sh
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json \
  --output_dir results/ss_flow_img_dit_1_3B_64_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"ss_latent\": \"datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"
```

This command trains the sparse-structure flow model on the **Objaverse-XL** dataset using the specified configuration file. Outputs are saved to `results/ss_flow_img_dit_1_3B_64_bf16`.

The dataset configuration includes:

* `base`: Root dataset directory.
* `ss_latent`: Directory containing precomputed sparse-structure latents.
* `render_cond`: Directory containing conditional rendering images.


The second- and third-stage flow models for shape and texture generation can be trained using the following configurations:

* Shape flow: `slat_flow_img2shape_dit_1_3B_512_bf16.json`
* Texture flow: `slat_flow_imgshape2tex_dit_1_3B_512_bf16.json`

Example commands:

```sh
# Shape flow model
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json \
  --output_dir results/slat_flow_img2shape_dit_1_3B_512_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"shape_latent\": \"datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"

# Texture flow model
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json \
  --output_dir results/slat_flow_imgshape2tex_dit_1_3B_512_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"shape_latent\": \"datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"pbr_latent\": \"datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_512\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"
```

Higher-resolution fine-tuning can be performed by updating the `finetune_ckpt` field in the following configuration files and adjusting the dataset paths accordingly:

* `slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json`
* `slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json`


## 🧩 Related Packages

TRELLIS.2 is built upon several specialized high-performance packages developed by our team:

*   **[O-Voxel](o-voxel):** 
    Core library handling the logic for converting between textured meshes and the O-Voxel representation, ensuring instant bidirectional transformation.
*   **[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM):** 
    Efficient sparse convolution implementation based on Triton, enabling rapid processing of sparse voxel structures.
*   **[CuMesh](https://github.com/JeffreyXiang/CuMesh):** 
    CUDA-accelerated mesh utilities used for high-speed post-processing, remeshing, decimation, and UV-unwrapping.


## ⚖️ License

This model and code are released under the **[MIT License](LICENSE)**.

Please note that certain dependencies operate under separate license terms:

- [**nvdiffrast**](https://github.com/NVlabs/nvdiffrast): Utilized for rendering generated 3D assets. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).

- [**nvdiffrec**](https://github.com/NVlabs/nvdiffrec): Implements the split-sum renderer for PBR materials. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt).

## 📚 Citation

If you find this model useful for your research, please cite our work:

```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
