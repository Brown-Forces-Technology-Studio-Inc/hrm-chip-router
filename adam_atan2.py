import torch
from torch.optim.optimizer import Optimizer

class AdamATan2(Optimizer):
    """Pure-PyTorch drop-in for adam-atan2 (eps-free Adam using atan2 update).
    a = 4/pi scales the bounded atan2 output to match Adam-like step magnitude."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0,
                 a=1.2732395447351628, b=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            a, b = group["a"], group["b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                num = exp_avg.div(bc1)
                den = exp_avg_sq.div(bc2).sqrt_().mul_(b)
                p.add_(torch.atan2(num, den), alpha=-lr * a)
        return loss
