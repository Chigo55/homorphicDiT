# utils/hooks.py
import torch
import warnings


def add_nan_hooks(module: torch.nn.Module):
    for name, p in module.named_parameters():
        if not p.requires_grad:          # ★ 학습 대상이 아니면 건너뛰기
            continue

        # -------- 그래디언트 검사 --------
        def _grad_check(grad, n=name):
            if torch.isnan(input=grad).any() or torch.isinf(input=grad).any():
                print(f"[GRAD NaN] {n}")

        # -------- 가중치 검사 (옵티머 step 직후) --------
        def _weight_check(grad, n=name):
            if torch.isnan(input=p.data).any() or torch.isinf(input=p.data).any():
                print(f"[WEIGHT NaN] {n}")

        p.register_hook(hook=_grad_check)
        p.register_hook(hook=_weight_check)
