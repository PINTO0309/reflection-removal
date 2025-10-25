from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from models import FeatureExtractorBase


def compute_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def compute_perceptual_loss(feature_extractor: FeatureExtractorBase,
                            pred: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
    pred_feats = feature_extractor.extract_features(pred, require_grad=True)
    with torch.no_grad():
        target_feats = feature_extractor.extract_features(target, require_grad=False)

    loss = 0.0
    for name, weight in feature_extractor.perceptual_layers:
        loss = loss + F.l1_loss(pred_feats[name], target_feats[name]) * weight
    return loss


def compute_gradient(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
    grady = img[:, :, :, 1:] - img[:, :, :, :-1]
    return gradx, grady


def compute_exclusion_loss(img1: torch.Tensor, img2: torch.Tensor, level: int = 3) -> torch.Tensor:
    gradx_losses = []
    grady_losses = []
    cur_img1, cur_img2 = img1, img2

    eps = 1e-6
    for _ in range(level):
        gradx1, grady1 = compute_gradient(cur_img1)
        gradx2, grady2 = compute_gradient(cur_img2)

        mean_gradx1 = gradx1.abs().mean(dim=(1, 2, 3), keepdim=True)
        mean_gradx2 = gradx2.abs().mean(dim=(1, 2, 3), keepdim=True)
        mean_grady1 = grady1.abs().mean(dim=(1, 2, 3), keepdim=True)
        mean_grady2 = grady2.abs().mean(dim=(1, 2, 3), keepdim=True)

        alphax = 2.0 * mean_gradx1 / (mean_gradx2 + eps)
        alphay = 2.0 * mean_grady1 / (mean_grady2 + eps)

        gradx1_s = torch.sigmoid(gradx1) * 2.0 - 1.0
        grady1_s = torch.sigmoid(grady1) * 2.0 - 1.0
        gradx2_s = torch.sigmoid(gradx2 * alphax) * 2.0 - 1.0
        grady2_s = torch.sigmoid(grady2 * alphay) * 2.0 - 1.0

        gx = ((gradx1_s.square()) * (gradx2_s.square())).mean(dim=(1, 2, 3))
        gy = ((grady1_s.square()) * (grady2_s.square())).mean(dim=(1, 2, 3))

        gradx_losses.append(torch.pow(gx + eps, 0.25))
        grady_losses.append(torch.pow(gy + eps, 0.25))

        cur_img1 = F.avg_pool2d(cur_img1, kernel_size=2, stride=2, padding=0)
        cur_img2 = F.avg_pool2d(cur_img2, kernel_size=2, stride=2, padding=0)

    if gradx_losses:
        gradx_term = torch.stack(gradx_losses, dim=0).sum(dim=0) / level
        grady_term = torch.stack(grady_losses, dim=0).sum(dim=0) / level
        loss = gradx_term.sum() + grady_term.sum()
    else:
        loss = torch.tensor(0.0, device=img1.device)
    return loss
