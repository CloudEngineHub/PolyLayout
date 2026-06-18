import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from moge.model.modules import ConvStack, DINOv2Encoder
from moge.utils.geometry_torch import normalized_view_plane_uv

from .base_model import BaseModel


class DINOv2(BaseModel):
    '''DINOv2-based feature extractor and edge detector.

    Architecture inspired by MoGe-2.
    @see https://github.com/microsoft/MoGe/blob/0286b495230a074aadf1c76cc5c679e943e5d1c6/moge/model/v2.py
    '''

    default_conf = {
        # Default settings mostly from 'moge-2-vits-normal' model
        'num_tokens_range': [1200, 3600],
        'encoder': {
            'backbone': 'dinov2_vits14',
            'intermediate_layers': [5, 11],
            'dim_out': 384
        },
        'feat_head': {
            'dim_in': [386, 2, 2, 2, 2],
            'dim_out': [128 + 1, None, 128 + 1, None, 32 + 1],
            'dim_res_blocks': [384, 256, 128, 64, 32],
            'num_res_blocks': [0, 1, 1, 1, 0],
            'res_block_in_norm': 'layer_norm',
            'res_block_hidden_norm': 'group_norm',
            'resamplers': ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear']
        },
        'edge_head': {
            'dim_in': [386, 2, 2, 2, 2],
            'dim_out': [1 + 1, None, 1 + 1, None, 1 + 1],
            'dim_res_blocks': [384, 256, 128, 64, 32],
            'num_res_blocks': [0, 1, 1, 1, 0],
            'res_block_in_norm': 'layer_norm',
            'res_block_hidden_norm': 'group_norm',
            'resamplers': ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear']
        },
        'num_tokens': None,
        'resolution_level': 9,
        'output_scales': [0, 2, 4],  # what scales to output
        'checkpointed': False,  # whether to use gradient checkpointing
    }

    def _init(self, conf):
        self.encoder = DINOv2Encoder(**OmegaConf.to_container(conf.encoder))
        self.encoder.init_weights()
        self.feat_head = ConvStack(**OmegaConf.to_container(conf.feat_head))
        self.edge_head = ConvStack(**OmegaConf.to_container(conf.edge_head))

        if conf.checkpointed:
            self.encoder.enable_gradient_checkpointing()
            self.feat_head.enable_gradient_checkpointing()
            self.edge_head.enable_gradient_checkpointing()

        self.scales = [2**s for s in conf.output_scales]

    def _forward(self, data):
        image = data['image']

        if image.ndim == 5:
            pred = [self._forward({'image': i}) for i in image]
            pred = {
                k: [torch.stack([p[k][i] for p in pred]) for i in range(len(v))]
                for k, v in pred[0].items()
                if v is not None
            }
            return pred

        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        num_tokens = self.conf.num_tokens
        if num_tokens is None:
            min_tokens, max_tokens = self.conf.num_tokens_range
            num_tokens = int(min_tokens + (self.conf.resolution_level / 9) * (max_tokens - min_tokens))

        aspect_ratio = img_w / img_h
        base_h, base_w = int((num_tokens / aspect_ratio) ** 0.5), int((num_tokens * aspect_ratio) ** 0.5)
        num_tokens = base_h * base_w

        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Heads
        feature_maps = self.feat_head(features)
        edge_maps = self.edge_head(features)

        pred = {'feature_maps': [], 'confidences': [], 'edge_maps': [], 'edge_confidences': []}
        for i, scale in zip(reversed(self.conf.output_scales), self.scales):
            out_h, out_w = img_h // scale, img_w // scale

            fmap = feature_maps[i]
            if fmap.shape[-2:] != (out_h, out_w):
                fmap = F.interpolate(fmap, (out_h, out_w), mode='bilinear', align_corners=False, antialias=False)
            pred['feature_maps'].append(fmap[:, :-1])
            pred['confidences'].append(torch.sigmoid(-fmap[:, -1:]))

            emap = edge_maps[i]
            if emap.shape[-2:] != (out_h, out_w):
                emap = F.interpolate(emap, (out_h, out_w), mode='bilinear', align_corners=False, antialias=False)
            pred['edge_maps'].append(torch.sigmoid(-emap[:, :-1]))
            pred['edge_confidences'].append(torch.sigmoid(-emap[:, -1:]))

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
