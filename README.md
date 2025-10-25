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
* To use the DINOv3 backbones (`dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`), place the provided checkpoints inside `./ckps/` (files named `dinov3_*_pretrain_lvd1689m-*.pth`). If the files are missing, they will be downloaded automatically through `torch.hub`.

## Dataset

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

`backbone: "vgg19", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16"`

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
--continue_training \
--data_syn_dir reflection-dataset/synthetic \
--data_real_dir reflection-dataset/real \
--backbone dinov3_vits16 \
--use_amp \
--exp_name dinov3_vits16
```

Checkpoints, intermediate predictions, `train.log`, and TensorBoard summaries (saved directly inside `runs/dinov3_vits16/`) are all stored under `runs/dinov3_vits16/`. Launch TensorBoard via:

```bash
tensorboard --logdir runs
```

## Arguments

`--exp_name`: experiment name. Training artifacts (checkpoints, `train.log`, image dumps) are stored under `./runs/<exp_name>/`, and inference outputs under `./test_results/<exp_name>/`.

`--data_syn_dir`: comma-separated list of synthetic dataset roots

`--data_real_dir`: comma-separated list of real dataset roots

`--save_model_freq`: frequency to save model and the output images

`--is_hyper`: whether to use hypercolumn features as input, all our trained models uses hypercolumn features as input

`--backbone`: feature extractor for hypercolumns and perceptual loss (`vgg19`, `dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`)

`--ckpt_dir`: directory where backbone checkpoints are searched (default `ckpts`)

`--test_only`: skip training and run inference only

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
