import cv2
import math
import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import convolve
from scipy.special import gamma
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)


class ImageQualityMetrics(nn.Module):
    def __init__(self, device="cuda", data_range=1.0):
        super().__init__()
        self.device_type = device

        # Reference-based metrics
        self.psnr = PeakSignalNoiseRatio(
            data_range=data_range).to(device=device)
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range).to(device=device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='squeeze').to(device=device)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.to(device=self.device_type)
        targets = targets.to(device=self.device_type)
        return {
            "PSNR": self.psnr(preds, targets).item(),
            "SSIM": self.ssim(preds, targets).item(),
            "LPIPS": self.lpips(preds, targets).squeeze().mean().item(),
        }

    def no_ref(self, preds: torch.Tensor):
        preds = preds.to(device=self.device_type)
        preds_np = preds.detach().cpu().numpy()
        preds_np = np.clip(a=preds_np, a_min=0, a_max=1)

        brisque_list = []
        for img in preds_np:
            # (C, H, W) -> (H, W, C)
            img_np = np.transpose(a=img, axes=(1, 2, 0))
            img_uint8 = (img_np * 255).astype(dtype=np.uint8)
            brisque_score = self._compute_brisque(img=img_uint8)
            brisque_list.append(brisque_score)

        return {
            "BRISQUE": float(np.mean(a=brisque_list)),
        }

    def full(self, preds, targets):
        ref_metrics = self.forward(preds=preds, targets=targets)
        no_ref_metrics = self.no_ref(preds=preds)
        return {**ref_metrics, **no_ref_metrics}

    def _compute_brisque(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        score = cv2.quality.QualityBRISQUE_compute(
            gray,
            "utils/files/brisque_model.yaml",
            "utils/files/brisque_range.yaml"
        )
        return score
