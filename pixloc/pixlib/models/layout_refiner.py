from copy import deepcopy

import omegaconf
import torch
from torch import Tensor
from torch.nn import functional as nnF

from . import get_model
from .two_view_refiner import TwoViewRefiner
from .utils import masked_mean
from ..geometry import Camera, Cuboid, Layout, Polygon, Pose
from ..geometry.losses import scaled_barron


ROTATIONAL_SYMMETRIES = torch.tensor(
    [
        [[-1.0, -0.0, -0.0], [-0.0, -0.0, -1.0], [-0.0, -1.0, -0.0]],
        [[-0.0, -1.0, -0.0], [-1.0, -0.0, -0.0], [-0.0, -0.0, -1.0]],
        [[-0.0, -0.0, -1.0], [-0.0, -1.0, -0.0], [-1.0, -0.0, -0.0]],
        [[-1.0, -0.0, -0.0], [-0.0, -1.0, -0.0], [0.0, 0.0, 1.0]],
        [[-0.0, -1.0, -0.0], [-0.0, -0.0, -1.0], [1.0, 0.0, 0.0]],
        [[-0.0, -0.0, -1.0], [-1.0, -0.0, -0.0], [0.0, 1.0, 0.0]],
        [[-1.0, -0.0, -0.0], [0.0, 1.0, 0.0], [-0.0, -0.0, -1.0]],
        [[-0.0, -1.0, -0.0], [0.0, 0.0, 1.0], [-1.0, -0.0, -0.0]],
        [[-0.0, -0.0, -1.0], [1.0, 0.0, 0.0], [-0.0, -1.0, -0.0]],
        [[-1.0, -0.0, -0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[-0.0, -1.0, -0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[-0.0, -0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [-0.0, -1.0, -0.0], [-0.0, -0.0, -1.0]],
        [[0.0, 1.0, 0.0], [-0.0, -0.0, -1.0], [-1.0, -0.0, -0.0]],
        [[0.0, 0.0, 1.0], [-1.0, -0.0, -0.0], [-0.0, -1.0, -0.0]],
        [[1.0, 0.0, 0.0], [-0.0, -0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, -0.0, -0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [-0.0, -1.0, -0.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-0.0, -1.0, -0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-0.0, -0.0, -1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, -0.0, -0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ]
)


class LayoutRefiner(TwoViewRefiner):
    default_conf = {
        'optimizer': {
            'name': 'layout_optimizer',
        },
        'pre_optimize_vp': False,
        'vp_opt_dist_thr': [2**(-i) for i in range(5)],
        'vp_opt_cam_margin': 2.5,
        'vp_opt_pixel_size': 1.0,
        'vp_opt_num_points_quadrant': None,
        'point_sampling': None,
        'num_points2D': 256,
        'guided_sampling_exp': 4.0,
        'split_layout_thr': 0.0,
        'split_layout_max_walls': 30,
    }
    required_data_keys = ['image', 'camera', 'T_w2cam', 'layout_init']

    def _init(self, conf):
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        assert hasattr(self.extractor, 'scales')

        Opt = get_model(conf.optimizer.name)
        if conf.duplicate_optimizer_per_scale:
            oconfs = [deepcopy(conf.optimizer) for _ in self.extractor.scales]
            for attr in ('feat_cost_scale',
                         'edge_cost_scale',
                         'vp_cost_scale',
                         'perimeter_cost_scale'):
                if not hasattr(conf.optimizer, attr):
                    continue
                cost_scale = getattr(conf.optimizer, attr)
                if not isinstance(cost_scale, float):
                    for s, oconf in zip(cost_scale, oconfs):
                        with omegaconf.read_write(oconf):
                            with omegaconf.open_dict(oconf):
                                setattr(oconf, attr, s)
            self.optimizer = torch.nn.ModuleList([Opt(c) for c in oconfs])
        else:
            self.optimizer = Opt(conf.optimizer)

    def optimize_vp(self, layout: Layout, T_w2cam: Pose, l2D: Tensor,
                    l2D_mask: Tensor) -> Layout:

        opt = self.optimizer
        if self.conf.duplicate_optimizer_per_scale:
            opt = opt[-1]
        lambda_ = opt.damping(layout)
        dist_thr_org = opt.vp_cost_fn.dist_thr

        for dist_thr in self.conf.vp_opt_dist_thr:
            opt.vp_cost_fn.dist_thr = dist_thr
            g, H = opt.build_system_vp(
                layout, T_w2cam, l2D, l2D_mask)
            dR, dd = opt.step(g, H, lambda_)
            layout = layout.__class__.from_Rd(dR @ layout.R, layout.d + dd)

        opt.vp_cost_fn.dist_thr = dist_thr_org

        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)
        if isinstance(layout, Cuboid):
            layout = layout.resize(cams_w, False, self.conf.vp_opt_cam_margin)
        elif self.conf.vp_opt_num_points_quadrant:
            layout = Polygon.from_circle(layout.R, cams_w,
                self.conf.vp_opt_num_points_quadrant, self.conf.vp_opt_cam_margin)
        else:
            layout = Polygon.from_concave_hull(layout.R, cams_w,
                self.conf.vp_opt_pixel_size, self.conf.vp_opt_cam_margin)
        return layout

    def sample_points(self, W: Tensor, camera: Camera) -> Tensor:
        if self.conf.point_sampling == 'guided':
            p2D = []
            exp = self.conf.guided_sampling_exp
            for w in W:
                idx = torch.multinomial(
                    w.reshape(w.shape[0], -1) ** exp, self.conf.num_points2D)
                p2D.append(
                    torch.stack([idx % w.shape[-1], idx // w.shape[-1]], axis=-1))
            p2D = torch.stack(p2D)
        elif self.conf.point_sampling == 'random':
            p2D = torch.rand(camera.shape + (self.conf.num_points2D, 2)).to(W)
            p2D *= camera.size.unsqueeze(-2)
        elif self.conf.point_sampling:
            raise ValueError(self.conf.point_sampling)

        visible = torch.ones(p2D.shape[:-1], dtype=bool, device=p2D.device)
        return p2D, visible

    def _forward(self, data):
        pred = self.extractor(data)
        pred['camera_pyr'] = [data['camera'].scale(1/s)
                              for s in self.extractor.scales]

        layout_init = data['layout_init']
        T_w2cam = data['T_w2cam']

        if self.conf.pre_optimize_vp:
            layout_init = self.optimize_vp(
                layout_init, T_w2cam, data['lines2D'], data['l2D_mask'])

        pred['layout_init'] = []
        pred['layout_opt'] = []
        for i in reversed(range(len(self.extractor.scales))):
            F = pred['feature_maps'][i]
            camera = pred['camera_pyr'][i]
            Wf = None
            if 'confidences' in pred and pred['confidences'] is not None:
                Wf = pred['confidences'][i]
            if self.conf.duplicate_optimizer_per_scale:
                opt = self.optimizer[i]
            else:
                opt = self.optimizer

            if self.conf.point_sampling:
                p2D, visible = self.sample_points(Wf, camera)
            else:
                p3D = data['points3D']
                p2D, visible = camera.world2image(T_w2cam * p3D)

            F_interp, mask, _ = opt.interpolator(F, p2D)
            mask &= visible

            Wf_interp = None
            if Wf is not None:
                Wf_interp, _, _ = opt.interpolator(Wf, p2D)

            E = pred['edge_maps'][i]

            We = None
            if 'edge_confidences' in pred and pred['edge_confidences'] is not None:
                We = pred['edge_confidences'][i]

            l2D = data.get('lines2D')
            l2D_mask = data.get('l2D_mask')

            if self.conf.normalize_features:
                F_interp = nnF.normalize(F_interp, dim=-1)
                F = nnF.normalize(F, dim=2)

            p2D_norm = camera.normalize(p2D)

            layout_opt, failed = opt(dict(
                layout_init=layout_init, T_w2cam=T_w2cam, camera=camera,
                p2D=p2D_norm, F=F, F_interp=F_interp, mask=mask, Wf=Wf,
                Wf_interp=Wf_interp, E=E, We=We, l2D=l2D, l2D_mask=l2D_mask))

            # Split polygon after optimization on coarse and medium scales
            if not self.training and i > 0 and self.conf.split_layout_thr:
                if isinstance(layout_opt, Cuboid):
                    layout_opt = Polygon.from_cuboid(layout_opt)
                layout_opt = layout_opt.split(
                    self.conf.split_layout_thr, self.conf.split_layout_max_walls)

            pred['layout_init'].append(layout_init)
            pred['layout_opt'].append(layout_opt)
            layout_init = layout_opt.detach()

        return pred

    def loss(self, pred, data):
        p3D = data['points3D']
        camera = data['camera']
        T_w2cam = data['T_w2cam']

        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)

        p2D, _ = camera.world2image(T_w2cam * p3D)
        p2D = camera.normalize(p2D)

        p3D = p3D.flatten(-3, -2).unsqueeze(-3)
        p2D_gt, visible_gt = camera.world2image(T_w2cam * p3D)

        p3D_mask = data['p3D_mask']
        p3D_mask = p3D_mask.flatten(-2, -1).unsqueeze(-2)

        def warp_error(layout: Layout):
            p3D_w, _ = layout.unsqueeze(-1).project(p2D, T_w2cam, return_jacobian=False)
            p3D_w = p3D_w.flatten(-3, -2).unsqueeze(-3)
            p2D_warped, visible_warped = camera.world2image(T_w2cam * p3D_w)

            if isinstance(layout, Polygon):
                p3D_proj, _ = layout.unsqueeze(-1).project(camera.normalize(p2D_warped), T_w2cam)
                visible_warped &= torch.all(torch.abs(p3D_proj - p3D_w) < 1e-3, dim=-1)

            num_cam, num_pt = p2D.shape[-3:-1]
            mask = ~torch.eye(num_cam, dtype=bool, device=layout.device)
            mask = mask.repeat(layout.shape + (1, 1))
            outside = ~layout.inside(cams_w)
            mask.masked_fill_(outside.unsqueeze(-1), False)
            mask.masked_fill_(outside.unsqueeze(-2), False)
            mask = mask.repeat_interleave(num_pt, -1)

            mask &= visible_gt & p3D_mask & visible_warped

            err = torch.sum((p2D_warped.flatten(-3, -2) - p2D_gt.flatten(-3, -2)) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0]/4
            err = masked_mean(err, mask.flatten(-2, -1), -1)
            return err

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}

        err_init = warp_error(pred['layout_init'][0])
        losses['warp_error/init'] = err_init

        for i, layout_opt in enumerate(pred['layout_opt']):
            err = warp_error(layout_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1-i]
            success = err < thresh
            losses[f'warp_error/{i}'] = err
            losses[f'success/{i}'] = success.float()
            losses[f'success/init/{i}'] = (err_init < thresh).float()
            losses['total'] += loss
        losses['warp_error'] = err
        losses['success'] = success.float()

        return losses

    def metrics(self, pred, data):
        layout_gt = data['layout_gt']
        layout_init = pred['layout_init'][0]

        p3D = data['points3D']
        p3D_mask = data['p3D_mask']
        camera = data['camera']
        T_w2cam = data['T_w2cam']

        p2D, _ = camera.world2image(T_w2cam * p3D)
        p2D = camera.normalize(p2D)

        @torch.no_grad()
        def rotation_error(layout1: Layout, layout2: Layout):
            R1, R2 = layout1.R.unsqueeze(1), layout2.R.unsqueeze(1)
            R = ROTATIONAL_SYMMETRIES.unsqueeze(0).to(layout1.device)
            dR = R1.transpose(-1, -2) @ (R @ R2)
            trace = torch.diagonal(dR, dim1=-1, dim2=-2).sum(-1)
            cos = torch.clamp((trace - 1) / 2, -1, 1)
            dr = torch.rad2deg(torch.acos(cos).abs())
            return torch.min(dr, dim=1).values

        @torch.no_grad()
        def distance_error(layout: Layout):
            p3D_proj, _ = layout.unsqueeze(-1).project(p2D, T_w2cam, return_jacobian=False)
            dist = torch.linalg.norm(p3D - p3D_proj, dim=-1)
            cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)
            inside = layout.inside(cams_w).unsqueeze(-1)
            mask = p3D_mask & inside
            err = masked_mean(dist.flatten(-2, -1), mask.flatten(-2, -1), -1)
            return err

        metrics = {}
        for i, layout_opt in enumerate(pred['layout_opt']):
            rot_err = rotation_error(layout_gt, layout_opt)
            metrics[f'rotation_error/{i}'] = rot_err
            dist_err = distance_error(layout_opt)
            metrics[f'3d_distance/{i}'] = dist_err
        metrics['rotation_error'] = rot_err
        metrics['3d_distance'] = dist_err

        rot_err_init = rotation_error(layout_gt, layout_init)
        metrics['rotation_error/init'] = rot_err_init
        dist_err_init = distance_error(layout_init)
        metrics['3d_distance/init'] = dist_err_init

        return metrics
