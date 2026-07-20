# Triangle-Field Super-Resolution Workflow

This document is the end-to-end checklist for training triangle-field
super-resolution models, starting from voxelized triangle fields, training or
selecting the VAE, encoding latents, and ending with the two SR variants:

- **Original / feature-space SR flow**: noisy high-res triangle-field features
  plus duplicated low-res conditioning, encoder predicts a latent, frozen VAE
  decoder predicts clean high-res features.
- **Latent SR flow**: noisy high-res VAE latents are decoded to feature space
  for model input, the encoder predicts clean high-res latents, and the frozen
  VAE decoder converts final latents back to triangle-field features.

## Split Rules

Do not use the root-level filtered directories as the train/test split.

- Training means **filtered train split only**.
- Evaluation means **filtered test split only**.
- The root-level directories such as `triangle_field_voxels_64` are payload
  stores. They contain the full filtered set, so the dataset must also be
  restricted by an instances file.

Use these split files:

```bash
ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k

# Filtered train subset used for training.
$ROOT/splits/train/instances.txt

# Filtered test subset used for evaluation.
$ROOT/splits/test/instances.txt
```

The expected filtered counts are:

- Train: `40364`
- Test: `4494`
- Full filtered train + test: `44858`

If a trainer prints `Total instances: 44858`, it is using the full filtered set
and is wrong for clean training/evaluation.

## End-To-End Flow

The clean workflow is:

1. Reuse or create triangle-field voxel payloads.
2. Train or choose a triangle-field VAE.
3. Optionally fine-tune that VAE at the target resolution.
4. Encode offline triangle-field latents from the selected VAE.
5. Train original feature-space SR flow.
6. Train latent-space SR flow.
7. Evaluate on filtered test only.

In this repo, steps 1-4 are usually already done for the current ObjXL4K
filtered experiment. Do not redo them unless you intentionally want a different
resolution, VAE checkpoint, latent dimension, decoder subdivision behavior, or
feature preprocessing.

## Existing Data

The ObjXL4K filtered triangle-field voxelizations already exist. Do not create
another copy unless you intentionally want to regenerate the voxelization.

Existing root-level payload directories:

```bash
$ROOT/triangle_field_voxels_32
$ROOT/triangle_field_voxels_64
$ROOT/triangle_field_voxels_128
$ROOT/triangle_field_voxels_256
$ROOT/triangle_field_voxels_512
```

The 64->128 SR experiments use:

```bash
$ROOT/triangle_field_voxels_64
$ROOT/triangle_field_voxels_128
```

Again, those are payload roots, not split roots. The split restriction is added
through `dataset.args.instances_path`.

## Choosing A VAE Resolution

The VAE is mostly convolutional/sparse and has behaved fairly resolution
agnostic in our experiments, but there are still practical choices:

- **Train at 256 first** if you want the cheaper/stabler base VAE. This gives
  fast iteration and lower VRAM pressure. A 256 VAE can often be evaluated on
  higher-resolution voxel payloads, but that should be treated as a generalizing
  checkpoint, not as guaranteed optimal at the new resolution.
- **Fine-tune at 512** if the downstream SR or reconstruction target is 512-ish
  quality, especially when inverse-area weighting and aux dropout matter. This
  is the path used by the current useful VAE.
- **Encode at the SR high resolution** for latent SR. For 64->128 latent SR, the
  high-res target latents are encoded from 128-resolution triangle fields.
- **Do not train/eval on full filtered payloads by accident.** VAE scripts may
  use split directories such as `$ROOT/splits/train/triangle_field_voxels_512`;
  SR scripts use root payload dirs plus `instances_path`.

Resolution options already present:

- `256`: cheaper base VAE / pred-subdiv experiments.
- `512`: current high-quality inverse-area + aux-drop VAE.
- `64` and `128`: used as SR low/high payloads.
- `32`: useful for cascade/base-resolution experiments.

### VAE Training Options

The main options in this repo are:

- **Standard 512 fine-tune from 256**:
  `configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_512_ft_objxl4k.json`
- **512 inverse-area + aux-drop fine-tune**:
  `configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_512_ft_objxl4k_invarea_auxdrop.json`
- **256 pred-subdiv latent-channel-64 experiment**:
  `configs/scvae/triangle_field_vae_next_dc_f16c64_fp16_predsubdiv_256_objxl4k.json`

Key option tradeoffs:

- **Inverse-area loss weighting** improves small-triangle emphasis by weighting
  voxels using inverse triangle area. This is useful for thin/fine geometry.
- **Aux dropout** zeros the non-`d_tri`/`d_vert` auxiliary channels during VAE
  fine-tuning so the encoder can later work when generated fields lack full aux
  features.
- **Frozen vs unfrozen decoder**: aux-drop fine-tuning can keep the decoder
  stable depending on config. For the currently useful checkpoint, treat the
  saved encoder/decoder pair as a matched VAE.
- **Latent channels 32 vs 64**: 32 is the current main path. 64 gives more
  capacity but changes downstream latent dimensionality and flow config.
- **`pred_subdiv=false`** uses GT/encoded sparse subdivision cache for decoding.
  This is the current main VAE path and is easiest for high-quality SR latents.
- **`pred_subdiv=true`** makes the decoder learn subdivision/support generation.
  This is useful if the downstream model must generate support, but it is a
  separate architecture/checkpoint family and not interchangeable with the main
  32-channel VAE latents.

## Train The Triangle-Field VAE

Current useful VAE path:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
sbatch train_triangle_field_vae_512_objxl4k_auxdrop_2gpu.sh
```

This uses:

```bash
configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_512_ft_objxl4k_invarea_auxdrop.json
```

and the 512 filtered train voxel payload:

```bash
$ROOT/splits/train/triangle_field_voxels_512
```

If training from scratch or doing a different VAE family, use the appropriate
script/config:

```bash
# 512 fine-tune from a 256 VAE.
sbatch train_triangle_field_vae_512_objxl4k.sh

# 256 pred-subdiv / latent-channel-64 experiment.
sbatch train_triangle_field_vae_256_objxl4k_predsubdiv_c64.sh
```

For SR, the important output is a matched encoder/decoder checkpoint pair. The
current useful pair is listed in the next section.

## Base VAE Checkpoint

Both SR variants use the inverse-area aux-drop triangle-field VAE decoder.

Useful VAE run:

```bash
$ROOT/outputs/triangle_field_vae_512_invarea_auxdrop_52039231
```

Useful checkpoint pair:

```bash
$ROOT/outputs/triangle_field_vae_512_invarea_auxdrop_52039231/ckpts/encoder_step0180000.pt
$ROOT/outputs/triangle_field_vae_512_invarea_auxdrop_52039231/ckpts/decoder_step0180000.pt
```

The latent SR model is initialized from the VAE encoder checkpoint above, not
from previous latent-SR checkpoints unless intentionally resuming a clean run.

## Current Best Setup

As of the latest comparisons, the best SR setup is the **non-high-KL all-res
VAE latent space** with the **input-style self-conditioning latent SR model**.

Best all-res VAE run:

```bash
$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251
```

Useful all-res VAE checkpoint:

```bash
$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251/ckpts/encoder_step0320000.pt
$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251/ckpts/decoder_step0320000.pt
```

Best latent SR flow run so far:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_52494005
```

Useful latent SR checkpoint:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_52494005/ckpts/encoder_ema0.9999_step0080000.pt
```

Best observed cascade inference setting:

```text
stage repeats: 3
steps per repeat: 11
CFG / guidance_strength: 1.0
base_guidance_strength: 0.0
conditioning augmentation: enabled
```

Existing encoded latents from this VAE:

```bash
$ROOT/triangle_field_latents/triangle_field_vae_allres_invarea_fullaux_52454251_step0320000_128
```

These are the current best 128-resolution triangle-field latents for 64->128
latent SR training/eval. Do not re-encode them unless the VAE checkpoint,
resolution, or filtered instance set intentionally changes.

Related existing latent payloads:

```bash
# High-KL all-res VAE latents.
$ROOT/triangle_field_latents/triangle_field_vae_allres_invarea_fullaux_highkl_52457336_step0320000_128

# Earlier inverse-area aux-drop VAE latents.
$ROOT/triangle_field_latents/triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_128
$ROOT/triangle_field_latents/triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_512

# DiT comparison / older VAE latent set.
$ROOT/triangle_field_latents/triangle_field_vae_51685536_step0100000_256

# Pred-subdiv c64 latent set.
$ROOT/triangle_field_latents/triangle_field_vae_256_predsubdiv_c64_52039264_step0150000_256
```

This model differs from the older latent SR flow by concatenating the current
latent/self-conditioning signal at the encoder input. The VAE latents are still
decoded through the frozen all-res VAE decoder; the extra input conditioning is
for the SR encoder, not a replacement for the decoder.

The training recipe for this setup is:

1. Train/fine-tune the all-res VAE on filtered train only.
2. Encode 128-resolution triangle-field latents with that all-res VAE.
3. Train the input-style latent SR model on filtered train only using those
   all-res VAE latents.
4. Evaluate on filtered test only, plus the no-train-duplicate filter when
   making comparison figures.

## Encode Triangle-Field Latents

Latent SR needs offline high-resolution latents from the selected VAE encoder.
For the current 64->128 latent SR setup, encode 128-resolution fields with the
512 inverse-area aux-drop VAE checkpoint:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export VAE_RUN=triangle_field_vae_512_invarea_auxdrop_52039231
export VAE_CKPT=step0180000

python data_toolkit/encode_triangle_field_latent.py \
  --root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --triangle_field_latent_root "$ROOT" \
  --resolution 128 \
  --model_root "$ROOT/outputs" \
  --enc_model "$VAE_RUN" \
  --ckpt "$VAE_CKPT" \
  --instances "$ROOT/splits/train/instances.txt" \
  --loader_workers 4 \
  --saver_workers 4
```

The output name is automatically:

```bash
$ROOT/triangle_field_latents/${VAE_RUN}_${VAE_CKPT}_128
```

For this VAE/checkpoint, that is:

```bash
$ROOT/triangle_field_latents/triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_128
```

If you need evaluation latents too, run the same command with:

```bash
--instances "$ROOT/splits/test/instances.txt"
```

The same latent payload directory can contain both train and test records; the
train/eval split is selected later through `instances_path` / `--split`.

Advanced option:

```bash
--allow_resolution_mismatch
```

Only use this for deliberate resolution-agnostic tests, e.g. encoding 128 or
256 fields with a VAE config trained at another resolution. For normal training,
prefer matching the encoded field resolution to the intended high-resolution SR
target.

## Offline Latents For Latent SR

Latent SR needs precomputed high-resolution triangle-field latents. The current
64->128 latent SR setup uses the 128-resolution latent payload:

```bash
$ROOT/triangle_field_latents/triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_128
```

This directory is also a full filtered payload root. During training, intersect
it with:

```bash
$ROOT/splits/train/instances.txt
```

During evaluation, intersect it with:

```bash
$ROOT/splits/test/instances.txt
```

## Original Feature-Space SR Flow

Config:

```bash
configs/gen/triangle_field_sr_flow_64to128_film_f16c32_fp16_objxl4k.json
```

Training script:

```bash
train_triangle_field_sr_flow_64to128.sh
```

The script now injects:

```json
"instances_path": "$ROOT/splits/train/instances.txt"
```

and starts with `--ckpt none` so a new run does not accidentally resume an old
SR checkpoint.

Run:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
sbatch train_triangle_field_sr_flow_64to128.sh
```

On successful initialization, the log should show:

```text
TriangleFieldSuperResolutionDataset
Total instances: 40364
Restricted to train/instances.txt: 40364
```

Historical run to know about:

```bash
$ROOT/outputs/triangle_field_sr_flow_64to128_film_52064461
```

This historical run was trained on the full filtered set (`44858`), so do not
use it for clean test-set claims. Its latest useful historical checkpoint was:

```bash
$ROOT/outputs/triangle_field_sr_flow_64to128_film_52064461/ckpts/encoder_step0140000.pt
```

## Latent SR Flow

Config:

```bash
configs/gen/triangle_field_latent_sr_flow_64to128_film_f16c32_fp16_objxl4k.json
```

Training script:

```bash
train_triangle_field_latent_sr_flow_64to128.sh
```

The script now injects:

```json
"instances_path": "$ROOT/splits/train/instances.txt"
```

and starts with `--ckpt none`, so the model initializes only from the VAE
checkpoint listed in the config:

```bash
$ROOT/outputs/triangle_field_vae_512_invarea_auxdrop_52039231/ckpts/encoder_step0180000.pt
```

Run:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
sbatch train_triangle_field_latent_sr_flow_64to128.sh
```

On successful initialization, the log should show:

```text
TriangleFieldLatentSuperResolutionDataset
Total instances: 40364
Restricted to train/instances.txt: 40364
Finetuning from:
  - encoder: .../triangle_field_vae_512_invarea_auxdrop_52039231/ckpts/encoder_step0180000.pt
```

Clean train-only run:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_film_52145106
```

This run trained on `40364` filtered-train samples and was cancelled by the
24-hour time limit after saving:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_film_52145106/ckpts/encoder_step0170000.pt
```

Historical run to know about:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_film_52102575
```

This historical run was trained on the full filtered set (`44858`), so do not
use it for clean test-set claims. Its useful historical checkpoint was:

```bash
$ROOT/outputs/triangle_field_latent_sr_flow_64to128_film_52102575/ckpts/encoder_step0070000.pt
```

## Evaluation

Evaluation should use the filtered test split:

```bash
$ROOT/splits/test/instances.txt
```

For latent SR, use:

```bash
python -u eval_triangle_field_latent_sr_flow.py \
  --run_dir "$ROOT/outputs/triangle_field_latent_sr_flow_64to128_film_52145106" \
  --root "$ROOT" \
  --split test \
  --ckpt latest \
  --guidance_strength 3.0 \
  --low_resolution 64 \
  --high_resolution 128 \
  --latent_name triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_128 \
  --steps 12 \
  --num_samples 16 \
  --batch_size 4 \
  --num_workers 0
```

The eval log should report:

```text
Total instances: 4494
Restricted to split/test: 4494
```

Use `cfg=3.0` as the default guidance strength unless explicitly testing a
different value.

For clean comparison figures, evaluation should also apply the test duplicate
removal filter:

```bash
$ROOT/metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv
```

The eval helpers now apply the filtered-test, no-train-duplicate, and triangle
filter intersection by default for `--split test`. The currently used fixed
16-example duplicate-filtered list is:

```bash
$ROOT/eval_instances_duplicate_filtered_test_n16.txt
```

### Cascaded Eval Without Repeat

This runs the normal cascade, where each stage is sampled once:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN=$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_52494005
export INST=$ROOT/eval_instances_duplicate_filtered_test_n16.txt

/home/koussa/scratch/envs/trellis2/bin/python eval_triangle_field_latent_sr_cascade.py \
  --run_dir "$RUN" \
  --root "$ROOT" \
  --ckpt 80000 \
  --ema_rate 0.9999 \
  --split test \
  --instances "$INST" \
  --num_samples 16 \
  --batch_size 1 \
  --steps 50 \
  --base_guidance_strength 0.0 \
  --guidance_strength 1.0 \
  --apply_conditioning_augmentation \
  --output_dir "$RUN/eval_cascade_step0080000_cfg1_aug_n16_dupfilter"
```

### Cascaded Eval With Repeat

This is the repeat-cascade strategy. It starts with unconditional 128, averages
that down to 64, then repeats each SR stage by feeding the generated high-res
field back through average downsampling:

```text
uncond 128 -> avg down to 64
64 -> 128 repeated xN
128 -> 256 repeated xN
256 -> 512 repeated xN
```

Run repeat-3:

```bash
cd /home/koussa/scratch/TRELLIS.2
export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN=$ROOT/outputs/triangle_field_latent_sr_flow_64to128_selfcond_input_allres_fullaux_52494005
export INST=$ROOT/eval_instances_duplicate_filtered_test_n16.txt

/home/koussa/scratch/envs/trellis2/bin/python eval_triangle_field_latent_sr_stage_repeat_cascade.py \
  --run_dir "$RUN" \
  --root "$ROOT" \
  --ckpt 80000 \
  --ema_rate 0.9999 \
  --split test \
  --instances "$INST" \
  --num_samples 16 \
  --batch_size 1 \
  --steps 11 \
  --base_guidance_strength 0.0 \
  --guidance_strength 1.0 \
  --apply_conditioning_augmentation \
  --stage_repeats 3 \
  --output_dir "$RUN/eval_repeat3_step0080000_cfg1_aug_n16_dupfilter"
```

This repeat-3 command is the current best observed inference recipe: it uses
conditioning augmentation, `11` steps per repeat, CFG `1.0`, and base CFG `0.0`.

For repeat-6, only change:

```bash
--stage_repeats 6
```

and use a separate output directory, for example:

```bash
--output_dir "$RUN/eval_repeat6_step0080000_cfg1_aug_n16_dupfilter"
```

Important eval details:

- Keep `--apply_conditioning_augmentation` on for SR cascade evals if the model
  was trained with conditioning augmentation.
- Use `--guidance_strength 1.0` for current comparison runs unless explicitly
  sweeping CFG.
- Use `--base_guidance_strength 0.0` for the initial unconditional/base path.
- Use the same `--instances` file when comparing methods side by side.

## Quick Sanity Checks

Before trusting a run:

```bash
rg -n "Total instances|Restricted|Finetuning from|Starting training" logs/<job>.out
```

Expected training lines:

```text
Total instances: 40364
Restricted to train/instances.txt: 40364
```

Expected evaluation lines:

```text
Total instances: 4494
Restricted to split/test: 4494
```

If either path reports `44858`, stop and fix the split restriction before using
the results.
