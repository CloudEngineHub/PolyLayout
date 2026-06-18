import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .base_dataset import BaseDataset, collate
from .layout_sampling import R_from_cams, sample_layout
from .line_segments import read_line_segments
from .view import read_view
from ..geometry import Camera, Cuboid, Pose
from ...settings import DATA_PATH


class Stanford2D3DS(BaseDataset):
    default_conf = {
        'dataset_dir': '2d3ds/',
        'image_subpath': '{}/persp/rgb/',
        'pose_subpath': '{}/persp/pose/',

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': False,
        'seed': 0,

        'read_line_segments': False,
        'max_num_line_segments': 100,

        'multi_room': False,
        'init_layout': None,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        if conf.init_layout is None:
            raise ValueError('The initial layout sampling strategy is required.')

        self.root = Path(DATA_PATH, conf.dataset_dir)
        self.conf = conf

        data_dir = Path(__file__).parent / '2d3ds'

        with open(data_dir / 'layouts_test.json') as f:
            self.cuboids = json.load(f)

        with open(data_dir / 'images_test.json') as f:
            self.image_tuples = json.load(f)

        if self.conf.multi_room:
            image_tuples = []
            for it in self.image_tuples:
                area, space = it['scene'].split(':')
                if not image_tuples or image_tuples[-1]['scene'] != area:
                    image_tuples.append({ 'scene': area, 'perspective_images': {} })
                image_tuples[-1]['perspective_images'][space] = it['perspective_images']
            self.image_tuples = image_tuples

    def _read_view(self, area, image_name, seed):
        image_dir = self.root / self.conf.image_subpath.format(area)
        image_path = image_dir / image_name

        pose_dir = self.root / self.conf.pose_subpath.format(area)
        pose_name = image_name.replace('rgb', 'pose').replace('.jpg', '.json')
        pose_path = pose_dir / pose_name
        with open(pose_path) as f:
            pose = json.load(f)

        K = np.array(pose['camera_k_matrix'])
        params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=params,
            width=pose['image_width'], height=pose['image_height']))

        Rt = np.array(pose['camera_rt_matrix'])
        T = Pose.from_Rt(Rt[:3, :3], Rt[:, 3])

        data = read_view(
            self.conf, image_path, camera, T, np.empty((0, 3)), np.empty(0))

        if self.conf.read_line_segments:
            lseg_path = image_dir / 'line_segments.npz'
            l2D, l2D_mask = read_line_segments(
                lseg_path, image_name, self.conf.max_num_line_segments, seed)
            data['lines2D'] = camera.normalize(l2D - 0.5).float()
            data['l2D_mask'] = torch.from_numpy(l2D_mask)

        return data

    def _read_room(self, area: str, space: str, image_list: List[str], seed: int):
        data = []
        for image in image_list:
            data.append(self._read_view(area, image, seed))
        data = collate(data)

        cuboid_gt = self.cuboids[f'{area}:{space}']
        R, t, s = (np.array(cuboid_gt[key]) for key in ('R', 't', 's'))
        data['layout_gt'] = Cuboid.from_Rts(R, t, s).float()

        data['scene'] = area
        data['room'] = space
        return data

    def __getitem__(self, idx):
        image_tuple = self.image_tuples[idx]
        scene = image_tuple['scene']
        seed = self.conf.seed + idx

        if self.conf.multi_room:
            data = []
            for space, image_list in image_tuple['perspective_images'].items():
                data.append(self._read_room(scene, space, image_list, seed))

            R = R_from_cams(torch.cat([d['T_w2cam'] for d in data]), seed)
            for d in data:
                d['layout_init'] = sample_layout(self.conf, d['T_w2cam'], seed, R=R).float()

            data = collate(data)
        else:
            area, space = scene.split(':')
            data = self._read_room(area, space, image_tuple['perspective_images'], seed)
            data['layout_init'] = sample_layout(self.conf, data['T_w2cam'], seed).float()

        return data

    def __len__(self):
        return len(self.image_tuples)
