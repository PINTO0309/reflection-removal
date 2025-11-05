# reflection-removal

A model for removing reflections using a single image.

## Setup

```bash
git clone https://github.com/PINTO0309/reflection-removal.git && cd reflection-removal
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

* All pretrained weights (VGG-19) are fetched automatically via `torchvision`, so no manual download is required.
* Additional backbones are supported:
  * **DINOv3 ViT-Tiny (`dinov3_vitt`)** — DEIMv2-finetuned weights stored at `ckpts/deimv2_dinov3_s_wholebody34.pth`. No `torch.hub` download is available, so keep this file locally.
  * **HGNetV2 (`hgnetv2`)** — DEIMv2-finetuned CNN backbone stored at `ckpts/deimv2_hgnetv2_n_wholebody34.pth`.
  * **DINOv3 standard variants** (`dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`) — place the provided checkpoints inside `./ckpts/` (files named `dinov3_*_pretrain_lvd1689m-*.pth`). If the files are missing, they will be downloaded automatically through `torch.hub`.

## Dataset
https://github.com/ceciliavision/perceptual-reflection-removal?tab=readme-ov-file#dataset

```
reflection-dataset/
├── real
│   ├── blended
│   │   ├── 100001.jpg
│   │   ├── 100002.jpg
│   │   ├── 100003.jpg
│   │   ├── 100004.jpg
│   │   └── 100005.jpg
│   └── transmission_layer
│       ├── 100001.jpg
│       ├── 100002.jpg
│       ├── 100003.jpg
│       ├── 100004.jpg
│       └── 100005.jpg
└── synthetic
    ├── reflection_layer
    │   ├── 100001.jpg
    │   ├── 100002.jpg
    │   ├── 100003.jpg
    │   ├── 100004.jpg
    │   └── 100005.jpg
    └── transmission_layer
        ├── 100001.jpg
        ├── 100002.jpg
        ├── 100003.jpg
        ├── 100004.jpg
        └── 100005.jpg
```

## Training

`backbone: "vgg19", "hgnetv2", "dinov3_vitt", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16"`

```bash
# Baseline
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--exp_name dinov3_vits16 \
--use_amp
```
```bash
# Initialized with DINOv3 default weights + Residual blocks
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--exp_name dinov3_vits16_residual \
--residual_skips \
--residual_init 0.1 \
--use_amp
```
```bash
# Initialized with pretrained weights + Residual blocks
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--exp_name dinov3_vits16_residual \
--ckpt_file ckpts/reflection_removal_dinov3_vits16.pt \
--residual_skips \
--residual_init 0.1 \
--use_amp
```
```bash
# Distributed Hypercolumn + Residual blocks
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--exp_name dinov3_vits16_disthyper_residual \
--use_distributed_hypercolumn \
--residual_skips \
--residual_init 0.1 \
--use_amp
```

### Distributed hypercolumns & channel reduction

- `--use_distributed_hypercolumn` replaces the full hypercolumn concat (raw backbone features + RGB) with a distributed projection: each backbone stage is first reduced by a learnable 1×1 convolution and the concatenated tensor is then compressed to 64 channels via a final 1×1. This greatly lowers memory usage while keeping the generator interface unchanged. Enable this flag whenever you plan to export compact ONNX graphs or train on high resolutions; checkpoints store the necessary projection weights.
- `--hypercolumn_channel_reduction_scale` controls how aggressively each stage is reduced. The reducer for a layer with `C` channels will emit `ceil(C / scale)` channels. The default (`4`) keeps roughly 25 % of the original channels per stage; higher values (e.g. `8`, `16`) shrink both parameters and FLOPs linearly, at the cost of fewer hypercolumn features. The final post-projection always outputs 64 channels, so downstream generator layers remain compatible across scales. When loading a checkpoint, the correct scale is inferred automatically, so you only need to pass this flag during training if you want a non-default compression ratio.
- Practical guidance:
  1. Start with `--use_distributed_hypercolumn --hypercolumn_channel_reduction_scale 4` for general training—it balances memory and fidelity.
  2. If you are memory-bound, increase the scale to `8` or `16`. Expect parameter counts and compute inside the hypercolumn projector to drop by roughly ½ and ¼ respectively when moving from 4→8→16.
  3. If you disable distributed hypercolumns, the generator reverts to concatenating fully-resolved backbone maps; this offers maximal information but is significantly heavier and may not match ONNX deployment paths.

To resume from a previous checkpoint inside `dinov3_vits16/`:

```bash
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--exp_name dinov3_vits16_residual \
--use_distributed_hypercolumn \
--residual_skips \
--residual_init 0.1 \
--use_amp \
--resume
```

|Value|Note|
|:-|:-|
|loss|The average `content loss` calculated by adding the `L1 coefficient + 0.2×perceptual + grad`. When actually updating the generator, this is multiplied by 100 and added to the next adv.|
|percep|The average perceptual loss, which measures distance in feature space (DINO/VGG).|
|grad|The average exclusion/gradient loss, which prevents the gradients of the transmitted image and the reflected image from overlapping.|
|adv|The average adversarial loss (BCE), which is used to make the classifier believe the image is "real."|
|feat_dist|Mean MSE between the student and projected teacher feature maps; reported only when feature distillation is enabled.|
|pix_dist|Mean L1 distance between the student outputs and the frozen teacher outputs (transmission, and reflection when available); reported only when pixel distillation is enabled.|

Therefore, the loss display is not the final total loss, but an indicator for checking the basic loss balance on the content side.

Distillation from `dinov3_vits16` to `dinov3_vitt` and fine-tuned backbone in DEIMv2 for students to learn.

```bash
# Without Residual blocks
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vitt \
--exp_name dinov3_vitt_distill \
--ckpt_dir ckpts \
--distill_teacher_backbone dinov3_vits16 \
--distill_teacher_checkpoint ckpts/reflection_removal_dinov3_vits16.pt \
--use_amp
```
```bash
# With Residual blocks
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vitt \
--exp_name dinov3_vitt_distill_residual \
--ckpt_dir ckpts \
--residual_skips \
--residual_init 0.1 \
--distill_teacher_backbone dinov3_vits16 \
--distill_teacher_checkpoint ckpts/reflection_removal_dinov3_vits16_residual.pt \
--use_amp
```
```bash
# With Distributed Hypercolumn + Residual blocks
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vitt \
--exp_name dinov3_vitt_distill_residual \
--ckpt_dir ckpts \
--use_distributed_hypercolumn \
--residual_skips \
--residual_init 0.1 \
--distill_teacher_backbone dinov3_vits16 \
--distill_teacher_checkpoint ckpts/reflection_removal_dinov3_vits16_residual.pt \
--use_amp

uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--exp_name dinov3_vits16_distill_disthyper_residual \
--ckpt_dir ckpts \
--use_distributed_hypercolumn \
--residual_skips \
--residual_init 0.1 \
--distill_teacher_backbone dinov3_vitb16 \
--distill_teacher_checkpoint ckpts/reflection_removal_dinov3_vitb16.pt \
--use_amp
```

## Interpreting the Loss Components

- **Content vs. adversarial balance**: For each generator update we optimise
  `content_loss = L1(reflection) + 0.2 × perceptual + grad`, and the actual objective that is backpropagated is `total_g = 100 × content_loss + adv`. A rise in `loss` usually means the reconstruction terms need attention; a rise in `adv` means the discriminator is currently winning.
- **Synthetic vs. real batches**: When training on synthetic pairs both the transmission and reflection branches contribute to the perceptual/L1 terms, so `loss` should generally be higher than during real batches (where reflection supervision is zero). Expect `grad` to be non-zero only for synthetic samples.
- **Healthy dynamics**: You want `loss`, `percep`, and `grad` to trend downward slowly while `adv` oscillates. If all climb together, the model is diverging. If `adv` collapses near 0 while others stagnate, the discriminator may be too weak—lower its learning rate or add more synthetic data. If `adv` stays very high but the other terms shrink, the discriminator is too strong—consider reducing the GAN weight (e.g., scaling the final `+ adv`) or adding label smoothing.
- **Practical monitoring**: Track the logged scalars in TensorBoard. Focus on the moving averages per epoch; transient spikes after checkpoint saves are normal. Compare checkpoints by running `--test_only` so you can visually confirm whether changes in the metrics translate to better separation.

## Per-Epoch Validation Dumps

After every training epoch the current generator runs inference on the images specified by `--test_dir`, and the outputs are written to `test_results/<exp_name>/epoch_<NNNN>/`. If `--test_dir` does not resolve to any images, the script falls back to a fixed set of up to 10 blended inputs gathered from `--data_real_dir`, so you still get a consistent visual trace of progress across epochs. Clean up these folders periodically if disk usage grows too large.

Each validation sample produces a directory numbered by processing order (e.g. `test_results/<exp_name>/epoch_0001/0001_<image_stem>/`) containing three PNGs:

- `input.png` — the raw blended input frame.
- `t_output.png` — the predicted transmission layer.
- `r_output.png` — the predicted reflection layer.

Checkpoints, intermediate predictions, `train.log`, and TensorBoard summaries (saved directly inside `runs/dinov3_vits16/`) are all stored under `runs/dinov3_vits16/`. Launch TensorBoard via:

```bash
tensorboard --logdir runs
```

## Arguments

`--exp_name`: experiment name. Training artifacts (checkpoints, `train.log`, image dumps) are stored under `./runs/<exp_name>/`, and inference outputs under `./test_results/<exp_name>/`.

`--data_syn_dir`: comma-separated list of synthetic dataset roots

`--data_real_dir`: comma-separated list of real dataset roots

`--save_model_freq`: frequency to save model and the output images

`--keep_checkpoint_history`: number of saved checkpoint epochs (`epoch_<NNNN>` folders under `runs/<exp_name>/`) to retain (0 keeps all)

`--backbone`: feature extractor for hypercolumns and perceptual loss (`vgg19`, `hgnetv2`, `dinov3_vitt`, `dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`). Hypercolumn features are always enabled; older runs that used `--is_hyper` now default to the same behaviour.

`--ckpt_dir`: directory where backbone checkpoints are searched (default `ckpts`)
`--ckpt_file`: optional generator checkpoint that seeds training; loads weights before the first epoch (ignored with `--resume`)

`--test_only`: skip training and run inference only

`--resume`: resume training from the last checkpoint in `runs/<exp_name>/`

`--use_amp`: enable torch.cuda.amp mixed precision (effective on CUDA devices)

`--epochs`: number of training epochs (default 100)

`--device`: device string such as `cuda:0` or `cpu`

## Testing

```bash
uv run python main.py \
--exp_name dinov3_vits16_test \
--test_only \
--backbone dinov3_vits16 \
--test_dir ./test_images
```

Make sure the `--backbone` flag matches the model that produced the checkpoint you are loading.

If `--test_only` is omitted, the script trains by default and writes checkpoints/metrics to `runs/<exp_name>/`.

Test outputs are written to `./test_results/<exp_name>/<image_name>/`.

## VITT (DEIMv2-S) backbone outputs -> generator inputs

1. `/generator/blocks.0/Add_1_output_0: float32[1,1601,192]` -> `float32[1,192,640,640]`
2. `/generator/blocks.4/Add_1_output_0: float32[1,1601,192]` -> `float32[1,192,640,640]`
3. `/generator/blocks.7/Add_1_output_0: float32[1,1601,192]` -> `float32[1,192,640,640]`
4. `/generator/blocks.11/Add_1_output_0: float32[1,1601,192]` -> `float32[1,192,640,640]`
5. `input: float32[1,3,640,640]`
6. `1 + 2 + 3 + 4 + 5` -> `/generator/Concat_14_output_0: float32[1,771,640,640]`

## ONNX Export

- Backbone + Head
  ```bash
  uv run python export_onnx.py \
  --checkpoint runs/dinov3_vitt/epoch_0001/checkpoint.pt \
  --output dinov3_vitt_gennerator_640x640_640x640.onnx \
  --backbone dinov3_vitt \
  --static_shape \
  --height 640 \
  --width 640 \
  --head_height 640 \
  --head_width 640

  uv run python export_onnx.py \
  --checkpoint runs/dinov3_vitt/epoch_0001/checkpoint.pt \
  --output dinov3_vitt_gennerator_640x640_320x320.onnx \
  --backbone dinov3_vitt \
  --static_shape \
  --height 640 \
  --width 640 \
  --head_height 320 \
  --head_width 320
  ```
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/5338c4fd-8086-4f5d-be68-3d90c1096c91" />

- Head only
  ```bash
  uv run python export_onnx.py \
  --checkpoint runs/dinov3_vitt/epoch_0001/checkpoint.pt \
  --output dinov3_vitt_gennerator_headonly_640x640_320x320.onnx \
  --backbone dinov3_vitt \
  --static_shape \
  --height 640 \
  --width 640 \
  --head_height 320 \
  --head_width 320 \
  --head_only
  ```
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/cff6fa45-ecc0-460b-a65b-182ae1e2bd13" />

## ONNX Inference
```bash
uv run python demo_reflection_removal.py \
--input test_images \
--output runs/test \
--model dinov3_vits16_gennerator_640x640_640x640.onnx \
--provider CUDAExecutionProvider
```

## Citation

If you use this repository in your research, please cite both the original method and this implementation:
```bibtex
@software{Hyodo_2025_reflection_removal,
  author    = {Katsuya Hyodo},
  title     = {reflection-removal: Reflection-Removal},
  year      = {2025},
  month     = {nov},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.17413165},
  url       = {https://github.com/PINTO0309/reflection-removal},
  abstract  = {A model for removing reflections using a single image.},
}
```

## Acknowledgments

- https://github.com/ceciliavision/perceptual-reflection-removal
  ```bibtex
  @inproceedings{zhang2018single,
    title = {Single Image Reflection Separation with Perceptual Losses},
    author = {Zhang, Xuaner and Ng, Ren and Chen, Qifeng},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2018}
  }
