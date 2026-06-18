from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import Cuboid, Layout, Polygon, Pose


def orthogonal_vector(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    m = (x != 0).argmax()
    n = m + 1 if m + 1 < x.shape[0] else 0
    y[n] = x[m]
    y[m] = -x[n]
    return y


def R_from_cams(T_w2cam: Pose, seed: Optional[int] = None) -> np.ndarray:
    '''Get rotation matrix from camera poses.

    Args:
        T_w2cam: camera poses.
        seed: random seed.
    Returns:
        R: rotation matrix.
    '''
    state = np.random.RandomState(seed)
    R_cam, _ = T_w2cam.numpy()

    down = np.mean(R_cam[:, 1], axis=0)
    z = -down
    z /= np.linalg.norm(z)
    y = orthogonal_vector(z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    a = state.uniform(0, 2 * np.pi)
    y = np.sin(a) * y + np.cos(a) * x
    x = np.cross(y, z)
    R = np.stack([x, y, z])
    return R.astype(R_cam.dtype)


def sample_cuboid_cams(
    T_w2cam: Pose,
    seed: Optional[int] = None,
    margin: float = 2.5,
    R: Optional[np.ndarray] = None,
) -> Cuboid:
    '''Sample cuboid based on known camera poses.

    Args:
        T_w2cam: camera poses.
        seed: random seed.
        margin: distance between camera centers and cuboid faces.
        R: cuboid orientation (optional).
    Returns:
        cuboid: sampled cuboid.
    '''
    if R is None:
        R = R_from_cams(T_w2cam, seed)

    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]

    cams = cams_w[..., 0] @ R.T
    d = np.r_[np.min(cams, axis=0) - margin, np.max(cams, axis=0) + margin]

    return Cuboid.from_Rd(R, d)


def sample_layout_gt(
    conf, layout_gt: Layout, T_w2cam: Pose, seed: Optional[int] = None,
) -> Layout:
    '''Sample layout by perturbing a ground truth layout.

    Args:
        conf: config.
        layout_gt: the ground truth layout.
        T_w2cam: camera poses.
        seed: random seed.
    Returns:
        layout: sampled layout.
    '''
    state = np.random.RandomState(seed)

    # https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
    axis = state.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = state.uniform(0, conf.init_layout_max_rot)
    dR = Rotation.from_rotvec(angle * axis).as_matrix()
    R = layout_gt.R @ dR

    if isinstance(layout_gt, Cuboid):
        dt = state.uniform(
            -conf.init_cuboid_max_trans, conf.init_cuboid_max_trans, 3)
        t = layout_gt.t + dt

        ds = state.uniform(
            conf.init_cuboid_max_grow[0], conf.init_cuboid_max_grow[1], 3)
        s = layout_gt.s + ds

        layout = Cuboid.from_Rts(R, t, s)
    else:
        dd = state.uniform(
            conf.init_polygon_max_shift[0], conf.init_polygon_max_shift[1], layout_gt.d.shape[0])
        d = layout_gt.d + dd
        layout = Polygon.from_Rd(R, d)

    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]
    return layout.resize(cams_w.squeeze(-1), True, conf.init_layout_cam_margin)


def sample_polygon_circle(
    T_w2cam: Pose,
    seed: Optional[int] = None,
    num_points_quadrant: int = 4,
    margin: float = 1.0,
    R: Optional[np.ndarray] = None,
) -> Polygon:
    '''Sample polygon based on known camera poses.

    Args:
        T_w2cam: camera poses.
        seed: random seed.
        num_points_quadrant: number of points per quadrant.
        margin: distance between camera centers and polygon faces.
        R: polygon orientation (optional).
    Returns:
        polygon: sampled polygon.
    '''
    if R is None:
        R = R_from_cams(T_w2cam, seed)
    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]
    return Polygon.from_circle(R, cams_w[..., 0], num_points_quadrant, margin)


def sample_polygon_hull(
    T_w2cam: Pose,
    seed: Optional[int] = None,
    pixel_size: float = 1.0,
    margin: float = 3.0,
    R: Optional[np.ndarray] = None,
) -> Polygon:
    '''Sample polygon based on known camera poses.

    Args:
        T_w2cam: camera poses.
        seed: random seed.
        pixel_size: pixel size for rasterization.
        margin: distance between camera centers and polygon faces.
        R: polygon orientation (optional).
    Returns:
        polygon: sampled polygon.
    '''
    if R is None:
        R = R_from_cams(T_w2cam, seed)
    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]
    return Polygon.from_concave_hull(R, cams_w[..., 0], pixel_size, margin)


def sample_layout(
    conf,
    T_w2cam: Pose,
    seed: Optional[int] = None,
    R: Optional[np.ndarray] = None,
    layout_gt: Optional[Layout] = None,
):
    '''Sample layout based on configuration.
    Args:
        conf: config.
        T_w2cam: camera poses.
        seed: random seed (optional).
        R: orientation (optional).
        layout_gt: ground truth layout (optional).

    Returns:
        layout_init: sampled layout.
    '''
    if conf.init_layout == 'ground_truth':
        return sample_layout_gt(conf, layout_gt, T_w2cam, seed)
    elif conf.init_layout == 'cuboid_cameras':
        return sample_cuboid_cams(T_w2cam, seed, R=R)
    elif conf.init_layout == 'cuboid_random':
        R = Rotation.random(random_state=np.random.RandomState(seed)).as_matrix()
        return sample_cuboid_cams(T_w2cam, seed, R=R)
    elif conf.init_layout == 'polygon_circle':
        return sample_polygon_circle(T_w2cam, seed, R=R)
    elif conf.init_layout == 'polygon_hull':
        return sample_polygon_hull(T_w2cam, seed, R=R)
    else:
        raise ValueError(conf.init_layout)
