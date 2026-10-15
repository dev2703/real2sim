"""Stateless, resumable per-update linear warmup followed by cosine decay."""

import math


def learning_rate(step, total_steps, peak, *, warmup_steps, start_factor, min_lr):
    if not 1 <= step <= total_steps or not 1 <= warmup_steps < total_steps:
        raise ValueError("schedule requires 1 <= step <= total and 1 <= warmup < total")
    if not (math.isfinite(peak) and math.isfinite(min_lr) and math.isfinite(start_factor)
            and 0 <= min_lr <= peak and 0 < start_factor <= 1):
        raise ValueError("invalid scheduler learning rates/factor")
    if step <= warmup_steps:
        fraction = (step - 1) / max(1, warmup_steps - 1)
        return peak * (start_factor + (1 - start_factor) * fraction)
    fraction = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (peak - min_lr) * (1 + math.cos(math.pi * fraction)) / 2
