import logging
from typing import Dict, Optional

import torch
from torch import Tensor

from .base_optimizer import BaseOptimizer
from .learned_optimizer import DampingNet
from ..geometry import Camera, Layout, Polygon, Pose
from ..geometry.layout_costs import FeaturemetricCost, EdgeCost, VanishingPointCost
from ..geometry.optimization import optimizer_step, so3exp_map
from ..geometry import losses  # noqa

logger = logging.getLogger(__name__)


class LayoutOptimizer(BaseOptimizer):
    '''Optimizer for cuboid or Manhattan polygon room layouts.'''
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        dd_stop_criteria=1e-4,
        feat_cost_scale=1.0,
        edge_cost_scale=0.0,
        edge_loss_fn='squared_loss',
        num_edge_points=40,
        vp_cost_scale=0.0,
        vp_loss_fn='squared_loss',
        vp_dist_thr=0.05,
        perimeter_cost_scale=0.0,
        perimeter_loss_fn='squared_loss',
        clamp_dd=0.0,
        expand_layout=True,
        expand_layout_margin=0.1,
        simplify_layout_thr=0.5,
        simplify_layout_dd_thr=None,
        simplify_layout_every=1,
        repair_layout_every=1,
    )

    def _init(self, conf):
        super()._init(conf)
        self.dampingnet = DampingNet(conf.damping, num_params=9)
        self.cost_fn = FeaturemetricCost(
            self.interpolator, normalize=conf.normalize_features)
        # self.loss_fn set in BaseOptimizer
        self.edge_cost_fn = EdgeCost(
            self.interpolator, self.conf.num_edge_points)
        self.edge_loss_fn = eval('losses.' + conf.edge_loss_fn)
        self.vp_cost_fn = VanishingPointCost(self.conf.vp_dist_thr)
        self.vp_loss_fn = eval('losses.' + conf.vp_loss_fn)
        self.perimeter_loss_fn = eval('losses.' + conf.perimeter_loss_fn)
        assert not conf.jacobi_scaling

    def local_frame(self, T_w2cam: Pose):
        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)
        t = cams_w.mean(axis=1)
        R = torch.eye(3).unsqueeze(0).expand(t.shape[0], -1, -1).to(t)
        return Pose.from_Rt(R, t)

    def damping(self, layout: Layout):
        lambda_ = self.dampingnet()

        if isinstance(layout, Polygon) and lambda_.shape[0] == 9:
            # Network was trained with Cuboid, remap to Polygon
            lambda_new = torch.empty(3 + layout.d.shape[-1], device=lambda_.device)
            lambda_new[:3] = lambda_[:3]  # rotation
            lambda_new[3] = lambda_[5]  # floor
            lambda_new[4] = lambda_[8]  # ceiling
            lambda_new[5:] = lambda_[[3, 4, 6, 7]].mean()  # walls
            lambda_ = lambda_new

        return lambda_

    def build_system_feat(self, layout: Layout, T_w2cam: Pose, camera: Camera,
                          p2D: Tensor, F: Tensor, F_interp: Tensor,
                          mask: Optional[Tensor], W: Optional[Tensor],
                          W_interp: Optional[Tensor]):

        res, valid, w_unc, J = self.cost_fn.residual_jacobian(
            layout, T_w2cam, camera, p2D, F, F_interp, W, W_interp)
        if mask is not None:
            valid &= mask.flatten(-2, -1).unsqueeze(-2)

        res = res.flatten(1, 2)
        valid = valid.flatten(1, 2)
        J = J.flatten(1, 2)

        # compute the cost and aggregate the weights
        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.loss_fn(cost)
        weights = self.conf.feat_cost_scale * w_loss * valid.float()
        if w_unc is not None:
            weights *= w_unc.flatten(1, 2)

        # solve the linear system
        g, H = self.build_system(J, res, weights)
        return g, H, valid

    def build_system_edge(self, layout: Layout, T_w2cam: Pose, camera: Camera,
                          E: Tensor, W: Tensor):

        res, valid, w_unc, J = self.edge_cost_fn.residual_jacobian(
            layout, T_w2cam, camera, E, W)

        res = res.flatten(1, 2)
        valid = valid.flatten(1, 2)
        J = J.flatten(1, 2)

        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.edge_loss_fn(cost)
        weights = self.conf.edge_cost_scale * w_loss * valid.float()
        if w_unc is not None:
            weights *= w_unc.flatten(1, 2)

        return self.build_system(J, res, weights)

    def build_system_vp(self, layout: Layout, T_w2cam: Pose, l2D: Tensor,
                        l2D_mask: Tensor):

        res, JR = self.vp_cost_fn.residual_jacobian(
            layout.unsqueeze(-1), T_w2cam, l2D)

        J = torch.nn.functional.pad(JR, (0, layout.d.shape[-1]))

        res = res.flatten(1, 2)
        valid = l2D_mask.flatten(1, 2)
        J = J.flatten(1, 2)

        cost = res**2
        cost, w_loss, _ = self.vp_loss_fn(cost)
        weights = self.conf.vp_cost_scale * w_loss * valid.float()

        return self.build_system(J.unsqueeze(-2), res.unsqueeze(-1), weights)

    def build_system_perimeter(self, layout: Layout):
        res, J = layout.perimeter(return_jacobian=True)
        res, J = res.unsqueeze(-1), J.unsqueeze(-2)

        cost = res**2
        cost, w_loss, _ = self.perimeter_loss_fn(cost)
        weights = self.conf.perimeter_cost_scale * w_loss

        return self.build_system(J.unsqueeze(-2), res.unsqueeze(-1), weights)

    def step(self, g: Tensor, H: Tensor, lambda_: Tensor,
             mask: Optional[Tensor] = None):

        delta = optimizer_step(g, H, lambda_, mask=mask)
        dw, dd = delta.split([3, delta.shape[-1] - 3], dim=-1)
        if self.conf.clamp_dd:
            # dd = dd.clamp(min=-self.conf.clamp_dd, max=self.conf.clamp_dd)
            dd = torch.tanh(2.0 / self.conf.clamp_dd * dd) * self.conf.clamp_dd
        dR = so3exp_map(dw)
        return dR, dd

    def expand_layout(self, layout: Layout, p3d: Tensor):
        return layout.resize(p3d, True, self.conf.expand_layout_margin)

    def simplify_layout(self, layout: Layout, dd: Tensor):
        if self.conf.simplify_layout_dd_thr is not None:
            keep = torch.abs(dd) > self.conf.simplify_layout_dd_thr
        else:
            keep = None
        return layout.simplify(self.conf.simplify_layout_thr, keep)

    def early_stop(self, **args):
        stop = False
        if not self.training and (args['i'] % 10) == 0:
            dR, dd, grad = args['dR'], args['dd'], args['grad']
            grad_norm = torch.norm(grad.detach(), dim=-1)
            small_grad = grad_norm < self.conf.grad_stop_criteria
            trace = torch.diagonal(dR, dim1=-1, dim2=-2).sum(-1)
            cos = torch.clamp((trace - 1) / 2, -1, 1)
            dr = torch.rad2deg(torch.acos(cos).abs())
            small_step = ((dr < self.conf.dR_stop_criteria)
                          & torch.all(dd.abs() < self.conf.dd_stop_criteria, dim=-1))
            if torch.all(small_step | small_grad):
                stop = True
        return stop

    def _forward(self, data: Dict):
        return self._run(
            data['layout_init'], data['T_w2cam'], data['camera'],
            data['p2D'], data['F'], data['F_interp'], data['mask'],
            data['Wf'], data['Wf_interp'], data['E'], data['We'],
            data['l2D'], data['l2D_mask'])

    def _run(self, layout_init: Layout, T_w2cam: Pose, camera: Camera,
             p2D: Tensor, F: Tensor, F_interp: Tensor, mask: Optional[Tensor],
             Wf: Optional[Tensor], Wf_interp: Optional[Tensor],
             E: Tensor, We: Optional[Tensor], l2D: Tensor, l2D_mask: Tensor):

        T_l2w = self.local_frame(T_w2cam)
        T_w2l = T_l2w.inv()
        T_w2cam = T_w2cam @ T_l2w.unsqueeze(-1)
        layout = layout_init @ T_l2w

        failed = torch.full(
            layout.shape, False, dtype=torch.bool, device=layout.device)
        lambda_ = self.damping(layout)
        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)

        for i in range(self.conf.num_iters):
            g, H = [], []

            if self.conf.feat_cost_scale > 0:
                g_feat, H_feat, valid = self.build_system_feat(
                    layout, T_w2cam, camera, p2D, F, F_interp, mask, Wf, Wf_interp)
                g.append(g_feat)
                H.append(H_feat)
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

            if self.conf.edge_cost_scale > 0:
                g_edge, H_edge = self.build_system_edge(
                    layout, T_w2cam, camera, E, We)
                g.append(g_edge)
                H.append(H_edge)

            if self.conf.vp_cost_scale > 0:
                g_vp, H_vp = self.build_system_vp(
                    layout, T_w2cam, l2D, l2D_mask)
                g.append(g_vp)
                H.append(H_vp)

            if isinstance(layout, Polygon) and self.conf.perimeter_cost_scale > 0:
                g_perimeter, H_perimeter = self.build_system_perimeter(layout)
                g.append(g_perimeter)
                H.append(H_perimeter)

            g = torch.stack(g).sum(0)
            H = torch.stack(H).sum(0)
            dR, dd = self.step(g, H, lambda_, mask=~failed)
            layout = layout.__class__.from_Rd(dR @ layout.R, layout.d + dd)

            if not self.training:
                if self.conf.expand_layout:
                    layout = self.expand_layout(layout, cams_w)

                if isinstance(layout, Polygon):
                    num_planes_before = layout.num_planes

                    if self.conf.simplify_layout_every and \
                        i % self.conf.simplify_layout_every == self.conf.simplify_layout_every - 1:
                        layout = self.simplify_layout(layout, dd)

                    if self.conf.repair_layout_every and \
                        i % self.conf.repair_layout_every == self.conf.repair_layout_every - 1:
                        layout = layout.make_valid()

                    if not torch.equal(layout.num_planes, num_planes_before):
                        lambda_ = self.damping(layout)

            self.log(i=i, layout_init=layout_init, layout=layout @ T_w2l, dR=dR, dd=dd)
            if self.early_stop(i=i, dR=dR, dd=dd, grad=g):
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return layout @ T_w2l, failed
