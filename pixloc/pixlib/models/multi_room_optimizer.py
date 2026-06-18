from typing import Optional

import torch
from torch import Tensor

from ..geometry import Cuboid, Layout, Pose
from ..geometry.optimization import optimizer_step, so3exp_map
from .layout_optimizer import LayoutOptimizer


class MultiRoomOptimizer(LayoutOptimizer):
    '''Optimizer for multi-room layouts with same orientation and
     optionally shared floor/ceiling height.'''
    default_conf = dict(
        shared_floor=False,
        shared_ceiling=False,
    )

    def create_d_mapping(self, layout: Layout):
        num_rooms, num_planes = layout.d.shape
        num_params = num_rooms * num_planes

        # The sharing of plane offsets is controlled by this table of size (num_rooms, num_planes).
        # Each entry indicates which parameter index the corresponding plane offset maps to.
        # If two or more entries are equal then those planes are optimized jointly.
        # An entry of -1 indicates that the plane is not present in the layout.
        d_mapping = torch.arange(num_params).reshape(num_rooms, num_planes).to(layout.device)

        if self.conf.shared_floor:
            idx = 2 if isinstance(layout, Cuboid) else 0
            d_mapping[:, idx] = num_params
            num_params += 1

        if self.conf.shared_ceiling:
            idx = 5 if isinstance(layout, Cuboid) else 1
            d_mapping[:, idx] = num_params
            num_params += 1

        # Sharing of walls can be implemented here similarly.

        mask = torch.isnan(layout.d)
        d_mapping[mask] = -1
        _, d_mapping[~mask] = torch.unique(d_mapping[~mask], return_inverse=True)
        self.d_mapping = d_mapping

        # Make shared plane offsets equal.
        for d in self.d_mapping_unique:
            idx = torch.where(self.d_mapping == d)
            layout.d[idx] = torch.mean(layout.d[idx])

    @property
    def d_mapping_unique(self):
        return torch.unique(self.d_mapping[self.d_mapping != -1])

    @property
    def num_params(self):
        return 3 + self.d_mapping_unique.numel()

    def local_frame(self, T_w2cam: Pose):
        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)
        t = cams_w.flatten(0, 1).mean(axis=0).squeeze(-1)
        R = torch.eye(3, device=t.device)
        return Pose.from_Rt(R, t)

    def damping(self, layout: Layout):
        self.create_d_mapping(layout)

        lambda_ = super().damping(layout)
        lambda_mapped = torch.zeros(self.num_params).to(lambda_)
        lambda_mapped[:3] = lambda_[:3]
        for d in self.d_mapping_unique:
            idx = torch.where(self.d_mapping == d)
            lambda_mapped[3 + d] = torch.mean(lambda_[3 + idx[1]])
        return lambda_mapped

    def build_system(self, J: Tensor, res: Tensor, weights: Tensor):
        num_params = self.num_params
        g = torch.zeros(num_params, dtype=J.dtype, device=J.device)
        H = torch.zeros((num_params, num_params), dtype=J.dtype, device=J.device)

        for i in range(J.shape[0]):  # Iterate over rooms
            g_, H_ = super().build_system(J[i], res[i], weights[i])
            d_mapping = self.d_mapping[i]
            d_mapping = d_mapping[d_mapping != -1]
            idx = torch.cat([torch.arange(3, device=J.device), 3 + d_mapping])
            g[idx] += g_[:len(idx)]
            H[idx[:, None], idx] += H_[:len(idx), :len(idx)]

        return g, H

    def step(self, g: Tensor, H: Tensor, lambda_: Tensor,
             mask: Optional[Tensor] = None):
        delta = optimizer_step(g, H, lambda_)
        dR = so3exp_map(delta[None, :3])
        dd = delta[self.d_mapping + 3]
        dd[self.d_mapping == -1] = 0.0
        if self.conf.clamp_dd:
            dd = torch.tanh(2.0 / self.conf.clamp_dd * dd) * self.conf.clamp_dd
        return dR, dd

    def expand_layout(self, layout: Layout, p3d: Tensor):
        layout_resized = super().expand_layout(layout, p3d)

        for d in self.d_mapping_unique:
            idx = torch.where(self.d_mapping == d)
            delta = layout_resized.d[idx] - layout.d[idx]

            if torch.any(delta < 0.0):
                assert torch.all(delta <= 0.0)
                layout_resized.d[idx] = torch.min(layout_resized.d[idx])
            elif torch.any(delta > 0.0):
                assert torch.all(delta >= 0.0)
                layout_resized.d[idx] = torch.max(layout_resized.d[idx])

        return layout_resized
