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
uv run python main.py \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--use_amp \
--exp_name dinov3_vits16
```

To resume from a previous checkpoint inside `dinov3_vits16/`:

```bash
uv run python main.py \
--resume \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--use_amp \
--exp_name dinov3_vits16
```
|Value|Note|
|:-|:-|
|loss|The average `content loss` calculated by adding the `L1 coefficient + 0.2×perceptual + grad`. When actually updating the generator, this is multiplied by 100 and added to the next adv.|
|percep|The average perceptual loss, which measures distance in feature space (DINO/VGG).|
|grad|The average exclusion/gradient loss, which prevents the gradients of the transmitted image and the reflected image from overlapping.|
|adv|The average adversarial loss (BCE), which is used to make the classifier believe the image is "real."|

Therefore, the loss display is not the final total loss, but an indicator for checking the basic loss balance on the content side.

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

## DEIMv2-S backbone output

`/model/backbone/Reshape_1_output_0: [1, 192, 40, 40]` or `/model/backbone/Resize_1`

## Citation

If you use this repository in your research, please cite both the original method and this implementation:
```bibtex
@misc{reflection_removal_repo,
  author = {PINTO0309},
  title = {reflection-removal},
  howpublished = {\url{https://github.com/PINTO0309/reflection-removal}},
  note = {Accessed: 2024-04-13}
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
