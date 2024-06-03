import torch
from torch import nn
from torchdiffeq import odeint
from functools import partial

from dit import DiT


class Sampler:
    def __init__(
        self,
        model: DiT,
        device="cuda",
        image_size=32,
        num_classes=10,
    ):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.num_classes = num_classes
        self.solver = partial(odeint, atol=1e-4, rtol=1e-4, method="euler")

    @torch.no_grad()
    def sample(self, batch_size, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y0 = torch.randn(batch_size, 3, self.image_size, self.image_size, device=self.device)
        traj = self.solver(
            func=lambda t, x: self.model.forward_with_cfg(x, t, y, cfg_scale),
            y0=y0,
            t=torch.linspace(0, 1, sample_steps)
        )
        if return_all_steps:
            return traj.clip(-1, 1) * 0.5 + 0.5
        return traj[-1].clip(-1, 1) * 0.5 + 0.5
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        y = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        y0 = torch.randn(self.num_classes * n_per_class, 3, self.image_size, self.image_size, device=self.device)
        traj = self.solver(
            func=lambda t, x: self.model.forward_with_cfg(x, t, y, cfg_scale),
            y0=y0,
            t=torch.linspace(0, 1, sample_steps)
        )
        traj = traj.clip(-1, 1) * 0.5 + 0.5
        if return_all_steps:
            return traj
        return traj[-1]