"""
SpikeFusion-GI 2.0 — Bidirectional Vision Mamba
==================================================
Selective SSM with both forward and backward scanning for 2D images.
This matches the real VMamba architecture (VSS Block).

Forward scan  : left → right, top → bottom (row-major)
Backward scan : right → left, bottom → top (reverse row-major)

Reference:
- Liu et al., "Vision Mamba: Efficient Visual Representation Learning with
  Bidirectional State Space Model" (arXiv:2401.09417)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScanModule(nn.Module):
    """Selective SSM scan (forward direction)."""

    def __init__(self, d_inner, d_state=16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        with torch.no_grad():
            self.A_log.copy_(-torch.rand(d_inner, d_state) * 0.9 - 0.1)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u, delta, B, C):
        """
        Args:
            u, delta: [B, L, d_inner]
            B, C:     [B, L, d_state]
        Returns:
            y: [B, L, d_inner]
        """
        batch, length, _ = u.shape
        device = u.device
        dtype = u.dtype

        A_neg = -torch.exp(self.A_log)                     # [d_in, d_state]
        a = torch.exp(delta.unsqueeze(-1) * A_neg)          # [B, L, d_in, d_s]
        b = delta.unsqueeze(-1) * B.unsqueeze(2)           # [B, L, d_in, d_s]

        h = torch.zeros(batch, self.d_inner, self.d_state,
                        dtype=dtype, device=device)
        ys = []
        for t in range(length):
            h = a[:, t] * h + b[:, t] * u[:, t].unsqueeze(-1)
            # C is shared across channels, so broadcast it over d_inner.
            y_t = (C[:, t].unsqueeze(1) * h).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        return y


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block (VSS Block style).
    Processes sequence in forward and backward directions,
    then fuses with a learnable gate.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.drop_path = drop_path

        # Norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection: x -> (x_conv, x_gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )

        # Forward SSM
        self.ssm_fwd = SelectiveScanModule(self.d_inner, d_state)
        # Backward SSM
        self.ssm_bwd = SelectiveScanModule(self.d_inner, d_state)

        # Projections for delta, B, C (shared for both directions)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Bidirectional fusion gate
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.d_inner * 2, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

        # Drop path (stochastic depth)
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _forward_direction(self, x, ssm, reverse=False):
        """Run one direction through SSM."""
        if reverse:
            x = torch.flip(x, dims=[1])
        ssm_params = self.x_proj(x)                               # [B, L, d_s*2+1]
        delta_raw, B, C = torch.split(
            ssm_params, [1, self.ssm_fwd.d_state, self.ssm_fwd.d_state], dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        y = ssm(x, delta, B, C)
        if reverse:
            y = torch.flip(y, dims=[1])
        return y

    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        Returns:
            y: [B, L, d_model]
        """
        residual = x
        x = self.norm(x)

        # Project and split
        xz = self.in_proj(x)
        x_conv, z = xz.chunk(2, dim=-1)

        # Causal conv
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_conv.size(-1)]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Forward + backward scans
        y_fwd = self._forward_direction(x_conv, self.ssm_fwd, reverse=False)
        y_bwd = self._forward_direction(x_conv, self.ssm_bwd, reverse=True)

        # Fusion gate
        y_cat = torch.cat([y_fwd, y_bwd], dim=-1)           # [B, L, 2*d_in]
        gate = self.fuse_gate(y_cat)                         # [B, L, d_in]
        y = gate * y_fwd + (1 - gate) * y_bwd

        # Gating with z branch
        y = y * F.silu(z)
        y = self.out_proj(y)

        # Drop path + residual
        y = residual + self.drop_path_layer(y)
        return y


class DropPath(nn.Module):
    """Stochastic depth (DropPath) regularisation."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class VisionMamba(nn.Module):
    """Bidirectional Vision Mamba stack."""

    def __init__(self, embed_dim=128, depth=3, d_state=16,
                 d_conv=4, expand=2, drop_path_rate=0.1):
        super().__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            BiMambaBlock(embed_dim, d_state, d_conv, expand, dp_rates[i])
            for i in range(depth)
        ])

    def forward(self, x):
        states = []
        for blk in self.blocks:
            x = blk(x)
            states.append(x.clone())
        return x, states
