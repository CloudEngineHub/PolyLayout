from collections import deque
from typing import List, Optional, Tuple, Union

import mapbox_earcut
import numpy as np
import shapely
import torch
from matplotlib.path import Path
from torch import Tensor

from .cuboid import Cuboid
from .contour_tracing import trace_contours
from .optimization import skew_symmetric
from .utils import to_homogeneous
from .wrappers import Pose, TensorWrapper, autobatch, autocast, nobatch


def area(poly: Union[np.ndarray, Tensor]) -> float:
    '''Calculate the signed area of a polygon.

    Clockwise polygons have negative area.

    Args:
        poly: (N, 2) array or tensor of polygon vertices.
    Returns:
        area: signed area of the polygon.
    '''
    area = 0.0
    for i in range(poly.shape[0]):
        j = (i + 1) % poly.shape[0]
        area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    return 0.5 * area


def closest_point(poly: Tensor, point: Tensor) -> Tuple[Tensor, int, float]:
    '''Find the closest point on a polygon to a given point.
    Args:
        poly: (N, 2) tensor of polygon vertices.
        point: (2,) point coordinates.
    Returns:
        closest: (2,) closest point on the polygon.
        idx: index such that (idx, (idx + 1) % N) is the closest edge.
        t: interpolation factor along the edge [0, 1].
    '''
    min_dist = float('inf')
    closest = None
    closest_idx = -1
    closest_t = 0.0
    N = poly.shape[0]
    for i in range(N):
        j = (i + 1) % N
        a, b = poly[i], poly[j]
        ab = b - a
        ap = point - a
        if torch.dot(ab, ab) > 0:
            t = torch.dot(ap, ab) / torch.dot(ab, ab)
        else:
            t = torch.tensor(0.0).to(poly)
        t = torch.clamp(t, 0, 1)
        proj = a + t * ab
        dist = torch.norm(point - proj)
        if dist < min_dist:
            min_dist = dist
            closest = proj
            closest_idx = i
            closest_t = t
    return closest, closest_idx, closest_t


def contains_points(poly: Tensor, points: Tensor) -> Tensor:
    '''Check if points are inside a polygon.

    Args:
        poly: (N, 2) tensor of polygon vertices.
        points: (M, 2) tensor of points to check.
    Returns:
        inside: (M,) boolean tensor indicating if points are inside the polygon.
    '''
    path = Path(poly.detach().cpu().numpy())
    inside = path.contains_points(points.detach().cpu().numpy())
    return torch.from_numpy(inside).to(points.device)


def to_manhattan(poly: Tensor, pixel_size: float) -> Tensor:
    '''Convert a general polygon to a Manhattan polygon through rasterization.

    Args:
        poly: (N, 2) tensor of polygon vertices.
        pixel_size: pixel size in rasterized image.
    Return:
        manhattan_poly: (M, 2) tensor of Manhattan polygon vertices.
    '''
    xmin, ymin = torch.min(poly, axis=0).values
    xmax, ymax = torch.max(poly, axis=0).values

    x_coords = torch.arange(xmin, xmax, pixel_size) + pixel_size / 2
    y_coords = torch.arange(ymin, ymax, pixel_size) + pixel_size / 2
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    mask = contains_points(poly, points)

    img = mask.reshape(len(y_coords), len(x_coords))
    contours = trace_contours(img)
    assert len(contours) > 0

    contour = torch.from_numpy(contours[0]).to(poly)
    manhattan_poly = (contour + 0.5) * pixel_size + torch.tensor([xmin, ymin])
    return manhattan_poly


class Polygon(TensorWrapper):
    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        '''Stack Polygons, padding plane offset vectors with NaNs as needed.'''
        max_planes = max(obj.num_planes.item() for obj in objects)
        for i in range(len(objects)):
            d = objects[i].d
            if d.shape[-1] != max_planes:
                if d.shape[-1] < max_planes:
                    pad = torch.full(d.shape[:-1] + (max_planes - d.shape[-1],), torch.nan).to(d)
                    d = torch.cat([d, pad], dim=-1)
                else: # d.shape[-1] > max_planes
                    d = d[..., :max_planes]
                objects[i] = Polygon.from_Rd(objects[i].R, d)
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    @autocast
    def from_Rd(cls, R: Tensor, d: Tensor):
        '''Polygon from a rotation matrix and plane offsets.
        Accepts numpy arrays or PyTorch tensors.

        R is the world-to-polygon rotation matrix (i.e. Xᴸ = R Xᴳ)
        d defines the polygon planes zᴸ = d[0] (floor)
                                     zᴸ = d[1], d[1] > d[0] (ceiling)
                                     xᴸ = d[2] (wall 1)
                                     yᴸ = d[3] (wall 2)
                                     xᴸ = d[4] (wall 3)
                                     yᴸ = d[5] (wall 4)
                                        .
                                        .
                                        .

        Walls are expected to be in counter-clockwise order.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            d: plane offsets vector with shape (..., N).
        '''
        assert R.shape[-2:] == (3, 3)
        assert d.shape[-1] >= 6 and d.shape[-1] % 2 == 0
        assert R.shape[:-2] == d.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), d], -1)
        return cls(data)

    @classmethod
    def from_cuboid(cls, cuboid: 'Cuboid'):
        '''Polygon from a Cuboid.'''
        R = cuboid.R
        d = cuboid.d[..., [2, 5, 0, 1, 3, 4]]
        return cls.from_Rd(R, d)

    @classmethod
    @autocast
    def from_poly2d(cls, R: Tensor, poly2d: Tensor, floor: Tensor, ceiling: Tensor):
        '''Polygon from a 2D Manhattan polygon and floor/ceiling heights.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            poly2d: 2D polygon vertices with shape (..., N, 2).
            floor: floor height with shape (...,).
            ceiling: ceiling height with shape (...,).
        '''
        if R.ndim > 2:
            raise NotImplementedError('Batched inputs not supported.')

        delta = torch.roll(poly2d, -1, dims=0) - poly2d
        if delta[::2, 0].abs().sum() > delta[1::2, 0].abs().sum():
            poly2d = torch.roll(poly2d, 1, dims=0)

        # Check Manhattan property
        assert torch.allclose(poly2d[::2, 0], poly2d[1::2, 0])
        assert torch.allclose(poly2d[::2, 1], torch.roll(poly2d[1::2, 1], 1))

        d_wall = torch.stack([p[..., i % 2] for i, p in enumerate(poly2d)], dim=-1).to(R)
        d = torch.cat([floor.unsqueeze(-1), ceiling.unsqueeze(-1), d_wall], dim=-1)
        return cls.from_Rd(R, d)

    @classmethod
    @autocast
    def from_concave_hull(cls, R: Tensor, points: Tensor, pixel_size: float, margin: float, ratio: float = 0.5):
        '''
        Create a polygon from the concave hull of a set of points.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            points: (..., N, 3) tensor of points in world coordinates.
            pixel_size: pixel size for rasterization.
            margin: distance between points and polygon faces.
            ratio: concave hull ratio parameter.

        Returns:
            Polygon instance.
        '''
        if R.ndim > 2: # Batched inputs
            polygons = []
            for i in range(R.shape[0]):
                polygons.append(cls.from_concave_hull(R[i], points[i], pixel_size, margin, ratio=ratio))
            return torch.stack(polygons)

        points_local = points @ R.transpose(-1, -2)
        xy = points_local[..., :2]
        multi_point = shapely.MultiPoint(xy.detach().cpu().numpy())

        hull = shapely.concave_hull(multi_point, ratio=ratio)
        if hull.geom_type == 'LineString':
            hull = hull.buffer(distance=0.01)  # Small buffer to convert to polygon
        assert hull.geom_type == 'Polygon' and len(hull.interiors) == 0

        hull = shapely.buffer(hull, distance=margin)
        while hull.geom_type in ('GeometryCollection', 'MultiPolygon'):
            idx = np.argmax([p.area for p in hull.geoms])
            hull = hull.geoms[idx]
        assert hull.geom_type == 'Polygon' # and len(hull.interiors) == 0

        poly = torch.from_numpy(np.stack(hull.exterior.xy, axis=-1)[:-1])
        manhattan_poly = to_manhattan(poly, pixel_size)

        z = points_local[..., 2]
        floor = torch.min(z, axis=-1).values - margin
        ceiling = torch.max(z, axis=-1).values + margin

        return cls.from_poly2d(R, manhattan_poly, floor, ceiling)

    @classmethod
    @autocast
    def from_circle(cls, R: Tensor, points: Tensor, num_points_quadrant: int, margin: float):
        '''
        Create a polygon that approximates a circle centered around the given points.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            points: (..., N, 3) tensor of points in world coordinates.
            num_points_quadrant: number of points per quadrant.
            margin: distance between points and polygon faces.

        Returns:
            Polygon instance.
        '''
        points_local = points @ R.transpose(-1, -2)
        xy = points_local[..., :2]

        center = xy.mean(dim=-2, keepdim=True)
        xy_centered = xy - center

        dist = torch.linalg.norm(xy_centered, dim=-1, keepdim=True)
        radius = torch.max(dist, dim=-2, keepdim=True).values + margin

        num_points = 4 * num_points_quadrant
        angles = torch.linspace(0, 2 * np.pi, num_points + 1, device=xy.device, dtype=xy.dtype)

        circle_pts = center + radius * torch.stack([torch.cos(angles), torch.sin(angles)], axis=1)
        d_wall = []
        axis = 0
        for idx in range(num_points):
            if idx % num_points_quadrant == 0:
                d_wall.append(circle_pts[..., idx, axis])
                axis = 1 - axis
            else:
                d_wall.append(circle_pts[..., idx, axis])
                d_wall.append(circle_pts[..., idx, 1 - axis])
        d_wall = torch.stack(d_wall, dim=-1).to(dtype=xy.dtype, device=xy.device)

        z = points_local[..., 2]
        floor = torch.min(z, axis=-1).values - margin
        ceiling = torch.max(z, axis=-1).values + margin

        d = torch.cat([floor.unsqueeze(-1), ceiling.unsqueeze(-1), d_wall], dim=-1)
        return cls.from_Rd(R, d)

    @property
    def R(self) -> Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def d(self) -> Tensor:
        '''Underlying plane offset vector with shape (..., P).'''
        return self._data[..., 9:]

    @property
    def num_planes(self) -> Tensor:
        '''Get the number of planes.'''
        return torch.sum(~torch.isnan(self.d), dim=-1)

    @property
    def num_walls(self) -> Tensor:
        '''Get the number of walls.'''
        return self.num_planes - 2

    def __matmul__(self, transform: 'Pose') -> 'Polygon':
        '''Apply transform.

        Xᶜ = RᵇᶜXᵇ (polygon)
        Xᵇ = RᵃᵇXᵃ + tᵃᵇ (transform)
        => Xᶜ = RᵇᶜRᵃᵇXᵃ + Rᵇᶜtᵃᵇ
        '''
        R = self.R @ transform.R
        t = (self.R @ transform.t.unsqueeze(-1)).squeeze(-1)
        d = self.d.clone()
        d[..., 0:2] -= t[..., 2:3]  # Floor/ceiling
        d[..., 2:] -= t[..., :2].repeat((1, ) * (t.dim() - 1) + ((d.shape[-1] - 2) // 2, ))  # Walls
        return self.__class__.from_Rd(R, d)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.d.numpy()

    @property
    @nobatch
    def corners(self) -> Tensor:
        '''Get the corners of the polygon in global coordinates (..., 2W, 3).'''
        d = self.d
        z0, z1 = d[..., 0], d[..., 1]
        d = d[..., 2:]
        p3d = []
        num_walls = self.num_walls

        for i in range(num_walls):
            x, y = d[..., i], d[..., (i + 1) % num_walls]
            if i % 2:
                x, y = y, x
            p3d.append(torch.stack([x, y, z0], dim=-1))
            p3d.append(torch.stack([x, y, z1], dim=-1))

        p3d = torch.stack(p3d, dim=0).to(dtype=self.dtype, device=self.device)
        return p3d @ self.R

    verts = corners

    @property
    @nobatch
    def edges(self) -> Tensor:
        '''Get the polygon edge indices (..., 3W, 2).'''
        edges = []
        for i in range(self.num_walls):
            j = (i + 1) % self.num_walls
            edges.append(torch.tensor([2 * i, 2 * i + 1]))
            edges.append(torch.tensor([2 * i, 2 * j]))
            edges.append(torch.tensor([2 * i + 1, 2 * j + 1]))
        return torch.stack(edges, dim=0).to(device=self.device)

    @property
    @nobatch
    def faces(self) -> Tensor:
        '''Get the polygon face indices (F, 3).'''
        faces = []
        num_walls = self.num_walls

        # Triangulate walls
        for idx in range(num_walls):
            idx_prev = (idx - 1) % num_walls
            faces.append(torch.tensor([2 * idx_prev, 2 * idx, 2 * idx_prev + 1]))
            faces.append(torch.tensor([2 * idx + 1, 2 * idx_prev + 1, 2 * idx]))

        # Triangulate the 2D polygon
        poly2d = self.poly2d.detach().cpu().numpy()
        rings = np.array([num_walls.item()])
        faces2d = mapbox_earcut.triangulate_float64(poly2d, rings)
        faces2d = faces2d.reshape(-1, 3).astype(np.int64)

        # Ceiling faces
        faces.extend([torch.from_numpy(2 * f + 1) for f in faces2d])

        # Floor faces
        faces2d = faces2d[:, ::-1]  # Reverse the order for floor
        faces.extend([torch.from_numpy(2 * f) for f in faces2d])

        faces = torch.stack(faces, dim=0).to(device=self.device)
        return faces

    @autobatch
    @autocast
    def inside(self, p3d: Tensor) -> Tensor:
        '''Check if points are inside the polygon.

        Args:
            p3d: points in world coordinate (..., N, 3).
        Returns:
            inside: boolean tensor indicating whether points are inside (..., N).
        '''
        p3d = p3d @ self.R.transpose(-1, -2)

        z = p3d[..., 2]
        d = self.d
        inside_z = (d[..., 0] <= z) & (z <= d[..., 1])

        inside_xy = contains_points(self.poly2d, p3d[..., :2])

        inside = inside_z & inside_xy
        return inside

    @autobatch
    @autocast
    def resize(self, p3d: Tensor, expand: bool, margin: float = 0.0) -> 'Polygon':
        '''Resize polygon so that points fit inside.

        Args:
            p3d: points in world coordinates (..., 3).
            expand: whether to only expand the polygon.
            margin: minimum distance between points and polygon faces.
        Returns:
            polygon: the resized polygon.
        '''
        assert expand, 'Resize only supports expanding the polygon.'
        p3d = p3d @ self.R.transpose(-1, -2)
        d = self.d.clone()

        # Floor/ceiling
        zmin = torch.min(p3d[..., 2], dim=-1).values - margin
        zmax = torch.max(p3d[..., 2], dim=-1).values + margin
        d[..., 0] = torch.minimum(d[..., 0], zmin)
        d[..., 1] = torch.maximum(d[..., 1], zmax)

        def move_wall(wall_idx, d_new):
            if d_new < self.d[2 + wall_idx]:
                d[2 + wall_idx] = torch.minimum(d[2 + wall_idx], d_new - margin)
            else:
                d[2 + wall_idx] = torch.maximum(d[2 + wall_idx], d_new + margin)

        # Walls
        poly2d = self.poly2d
        inside = contains_points(poly2d, p3d[..., :2])
        num_walls = self.num_walls
        for idx in range(inside.shape[0]):
            if inside[idx]:
                continue

            _, wall_idx, t = closest_point(poly2d, p3d[idx, :2])
            wall_idx = (wall_idx + 1) % num_walls

            move_wall(wall_idx, p3d[idx, wall_idx % 2])

            if t == 0:
                prev_idx = (wall_idx - 1) % num_walls
                move_wall(prev_idx, p3d[idx, prev_idx % 2])
            elif t == 1:
                next_idx = (wall_idx + 1) % num_walls
                move_wall(next_idx, p3d[idx, next_idx % 2])

        polygon = Polygon.from_Rd(self.R, d)
        return polygon

    @property
    @nobatch
    def poly2d(self) -> Tensor:
        '''Get the 2D (xy) polygon in local coordinates (W, 2).'''
        d = self.d[..., 2:]
        p2d = []
        num_walls = self.num_walls
        for i in range(num_walls):
            x, y = d[..., i], d[..., (i + 1) % num_walls]
            if i % 2:
                x, y = y, x
            p2d.append(torch.stack([x, y], dim=-1))
        return torch.stack(p2d, dim=0).to(dtype=self.dtype, device=self.device)

    def sample_edges(
        self, num_points: int, return_jacobian: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        '''Sample points on the edges of the polygon.

        Args:
            num_points: number of points to sample on each edge.
            return_jacobian: whether to return the jacobian.
        Returns:
            p3d_w: the sampled points in world coordinates (..., 3*W*N, 3).
            valid: boolean tensor indicating whether points are valid (..., 3*W*N).
            J: the jacobian of p3d_w with respect to R and d (..., 3*W*N, 3, 3+P).
        '''
        d = self.d
        z0, z1 = d[..., 0], d[..., 1]
        d = d[..., 2:]
        num_walls = self.num_walls
        t = (
            torch.arange(num_points, device=self.device, dtype=self.dtype) + 0.5
        ) / num_points
        t_expanded = t
        for _ in range(self.dim()):
            t_expanded = t_expanded.unsqueeze(0)
        t_expanded = t_expanded.expand(self.shape + (-1, )).unsqueeze(-1)

        p3d = []
        valid = []
        if return_jacobian:
            Jd = []
        else:
            J = None

        for idx in range(torch.max(num_walls)):
            d_cur = d[..., idx]
            idx_prev = (idx - 1) % num_walls
            idx_next = (idx + 1) % num_walls
            d_prev = torch.gather(d, -1, idx_prev.unsqueeze(-1)).squeeze(-1)
            d_next = torch.gather(d, -1, idx_next.unsqueeze(-1)).squeeze(-1)

            x0, y0 = d_cur, d_prev
            x1, y1 = d_cur, d_next
            if idx % 2:
                x0, y0 = y0, x0
                x1, y1 = y1, x1

            f0 = torch.stack([x0, y0, z0], dim=-1).unsqueeze(-2)
            c0 = torch.stack([x0, y0, z1], dim=-1).unsqueeze(-2)
            f1 = torch.stack([x1, y1, z0], dim=-1).unsqueeze(-2)
            c1 = torch.stack([x1, y1, z1], dim=-1).unsqueeze(-2)

            p3d.append(torch.lerp(f0, c0, t.unsqueeze(-1)))
            p3d.append(torch.lerp(f0, f1, t.unsqueeze(-1)))
            p3d.append(torch.lerp(c0, c1, t.unsqueeze(-1)))
            valid.append(~torch.isnan(d_cur).unsqueeze(-1).repeat_interleave(3 * num_points, dim=-1))

            if return_jacobian:
                size_idx = self.shape + (num_points, -1)
                idx_prev = idx_prev.unsqueeze(-1).unsqueeze(-1).expand(size_idx)
                idx_next = idx_next.unsqueeze(-1).unsqueeze(-1).expand(size_idx)
                size_J = self.shape + (num_points, 3, self.d.shape[-1])

                Jd1 = torch.zeros(size_J, device=self.device, dtype=self.dtype)
                Jd1[..., 2, 0] = 1 - t
                Jd1[..., 2, 1] = t
                Jd1[..., (idx + 1) % 2, :].scatter_(-1, 2 + idx_prev, 1)
                Jd1[..., idx % 2, 2 + idx] = 1
                Jd.append(Jd1)

                Jd2 = torch.zeros(size_J, device=self.device, dtype=self.dtype)
                Jd2[..., (idx + 1) % 2, :].scatter_(-1, 2 + idx_prev, 1 - t_expanded)
                Jd2[..., (idx + 1) % 2, :].scatter_(-1, 2 + idx_next, t_expanded)
                Jd2[..., idx % 2, 2 + idx] = 1
                Jd2[..., 2, 0] = 1
                Jd.append(Jd2)

                Jd3 = torch.zeros(size_J, device=self.device, dtype=self.dtype)
                Jd3[..., (idx + 1) % 2, :].scatter_(-1, 2 + idx_prev, 1 - t_expanded)
                Jd3[..., (idx + 1) % 2, :].scatter_(-1, 2 + idx_next, t_expanded)
                Jd3[..., idx % 2, 2 + idx] = 1
                Jd3[..., 2, 1] = 1
                Jd.append(Jd3)

        p3d = torch.cat(p3d, dim=-2)
        valid = torch.cat(valid, dim=-1)
        p3d[~valid] = 0.0
        p3d_w = p3d @ self.R

        if return_jacobian:
            Rt = self.R.unsqueeze(-3).transpose(-1, -2)
            JR = skew_symmetric(p3d_w) @ Rt
            Jd = torch.cat(Jd, dim=-3)
            J = torch.cat([JR, Rt @ Jd], dim=-1)

        return p3d_w, valid, J

    def project(
        self, p2d: Tensor, T_w2cam: 'Pose', return_jacobian: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        '''Project image points onto the faces of the polygon.

        It is assumed that the camera is inside the polygon, so every 2D point
        will have a corresponding 3D point.

        Distortion is not supported, and thus there is no Camera passed to
        the function.

        Args:
            p2d: normalized image points (..., 2).
            T_w2cam: camera pose.
            return_jacobian: whether to return the jacobian.
        Returns:
            p3d_w: the projected 3D points in world coordinates (..., 3).
            J: the jacobian of p3d_w with respect to R and d (..., 3, 3+P).
        '''
        R_w2cam, t_w2cam = T_w2cam.R, T_w2cam.t.unsqueeze(-1)
        R, d = self.R, self.d
        num_walls = self.num_walls
        eps = 1e-8

        n3d = torch.cat([
            torch.tensor([0, 0, 1]).repeat(2, 1),
            torch.tensor([[1, 0, 0], [0, 1, 0]]).repeat(self.d.shape[-1] - 2, 1),
        ], dim=0).to(p2d)
        n3d_w = (n3d @ R).unsqueeze(-1)

        p3d_cam = to_homogeneous(p2d)
        p3d_w = torch.zeros_like(p3d_cam)
        λ_min = torch.full(p2d.shape[:-1], torch.inf).to(p2d)
        if return_jacobian:
            J = torch.zeros(p2d.shape[:-1] + (3, 3 + self.d.shape[-1])).to(p2d)
        else:
            J = None

        for i in range(self.d.shape[-1]):
            # For normalized 2D points x and 3D points X we have
            #  λx = RX + t <=> X = λRᵀx - Rᵀt
            # We solve for λ such that the 3D points lie on the plane
            #  nᵀX - d = 0
            #  => λ = (nᵀRᵀt + d) / nᵀRᵀx
            n3d_wi = n3d_w[..., i, :, :]

            ntRt = n3d_wi.transpose(-2, -1) @ R_w2cam.transpose(-2, -1)
            λ = (ntRt @ t_w2cam + torch.nan_to_num(d[..., i, None, None])) / (
                p3d_cam @ ntRt.transpose(-2, -1) + eps
            )
            p3d_wi = (λ * p3d_cam - t_w2cam.transpose(-1, -2)) @ R_w2cam

            λ = λ.squeeze(-1)
            valid = (0 < λ) & (λ < λ_min)
            valid &= ~torch.isnan(d[..., i]).unsqueeze(-1)

            if i > 1:  # Only for walls
                p3d = p3d_wi @ R.transpose(-2, -1)
                d_prev = torch.gather(d, -1, (2 + (i - 2 - 1) % num_walls).unsqueeze(-1))
                d_next = torch.gather(d, -1, (2 + (i - 2 + 1) % num_walls).unsqueeze(-1))
                lo, hi = torch.minimum(d_prev, d_next), torch.maximum(d_prev, d_next)
                axis = (i + 1) % 2
                valid &= (lo <= p3d[..., axis]) & (p3d[..., axis] <= hi)

            λ_min = torch.where(valid, λ, λ_min)
            p3d_w[valid] = p3d_wi[valid]

            if return_jacobian:
                J_p3d_λ = (p3d_cam @ R_w2cam).unsqueeze(-1)

                J_λ_d = 1.0 / (p3d_cam @ ntRt.transpose(-2, -1) + eps)
                Rn = R_w2cam @ n3d_wi
                J_λ_n = (
                    p3d_cam @ Rn * t_w2cam.transpose(-2, -1)
                    - (Rn.transpose(-2, -1) @ t_w2cam + torch.nan_to_num(d[..., i, None, None]))
                    * p3d_cam
                ) / (p3d_cam @ Rn + eps) ** 2
                J_λ_nd = torch.cat([J_λ_n @ R_w2cam, J_λ_d], dim=-1).unsqueeze(
                    -2
                )

                J_n_R = skew_symmetric(n3d_wi[..., 0]) @ R.transpose(-2, -1)
                J_n_d = torch.zeros(self.shape + (3, self.d.shape[-1])).to(p2d)
                J_n_Rd = torch.cat([J_n_R, J_n_d], dim=-1)
                J_d_R = torch.zeros(self.shape + (1, 3)).to(p2d)
                J_d_d = torch.zeros(self.shape + (1, self.d.shape[-1])).to(p2d)
                J_d_d[..., 0, i] = 1.0
                J_d_Rd = torch.cat([J_d_R, J_d_d], dim=-1)
                J_nd_Rd = torch.cat([J_n_Rd, J_d_Rd], dim=-2).unsqueeze(-3)

                J_p3d_Rd = J_p3d_λ @ J_λ_nd @ J_nd_Rd
                J[valid] = J_p3d_Rd[valid]

        return p3d_w, J

    def perimeter(self, return_jacobian: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        '''Compute the perimeter of the polygon.

        Args:
            return_jacobian: Whether to return the jacobian.
        Returns:
            perimeter: The perimeter of the polygon (...).
            J: The jacobian with respect to the polygon parameters (..., 3+P).
        '''
        num_walls = self.num_walls
        d = self.d[..., 2:]
        perimeter = torch.zeros(self.shape, dtype=d.dtype, device=d.device)

        if return_jacobian:
            J = torch.zeros(self.shape + (3 + self.d.shape[-1], )).to(perimeter)
        else:
            J = None

        for idx in range(torch.max(num_walls)):
            idx_prev = ((idx - 1) % num_walls).unsqueeze(-1)
            idx_next = ((idx + 1) % num_walls).unsqueeze(-1)

            d_cur = d[..., idx]
            d_prev = torch.gather(d, -1, idx_prev).squeeze(-1)
            d_next = torch.gather(d, -1, idx_next).squeeze(-1)

            delta = d_next - d_prev
            delta.masked_fill_(torch.isnan(d_cur), 0.0)
            perimeter += torch.abs(delta)

            if return_jacobian:
                delta = delta.unsqueeze(-1)
                J.scatter_reduce_(-1, 3 + 2 + idx_prev, -torch.sign(delta), 'sum')
                J.scatter_reduce_(-1, 3 + 2 + idx_next, torch.sign(delta), 'sum')

        return perimeter, J

    @autobatch
    def simplify(self, thr: float, keep: Optional[Tensor] = None) -> 'Polygon':
        '''Simplify layout using a variant of the Visvalingam-Whyatt algorithm
        for Manhattan polygons.

        Args:
            thr (float): The area threshold below which wall pairs are removed.
            keep (Optional[Tensor]): Boolean tensor of shape (W,) indicating
                which walls should be kept.
        Returns:
            polygon: Simplified polygon.
        '''
        class Wall:
            def __init__(self, d, idx, prev=None, next=None):
                self.d = d
                self.idx = idx
                self.prev = prev
                self.next = next
                self.removed = False

            def width(self):
                return torch.abs(self.next.d - self.prev.d)

        d = self.d[..., 2:]
        num_walls = self.num_walls

        # Create linked list of walls
        walls = [Wall(d[idx], idx) for idx in range(num_walls)]
        for idx in range(num_walls):
            idx_next = (idx + 1) % num_walls
            walls[idx].next = walls[idx_next]
            walls[idx_next].prev = walls[idx]

        # Simplify
        while num_walls > 4:
            min_area, wall1, wall2 = float('inf'), None, None
            for wall in walls:
                if wall.removed or wall.next.removed:
                    continue
                if keep is not None and (keep[wall.idx] or keep[wall.next.idx]):
                    continue
                area = wall.width() * wall.next.width()
                if area < min_area:
                    min_area = area
                    wall1, wall2 = wall, wall.next

            if min_area >= thr:
                break

            # Remove wall pair
            wall1.removed = wall2.removed = True
            num_walls -= 2

            # Update links
            wall1.prev.next = wall2.next
            wall2.next.prev = wall1.prev

        # Create new polygon
        walls = [w for w in walls if not w.removed]
        d_new = torch.tensor([w.d for w in walls], dtype=self.dtype, device=self.device)
        if walls[0].idx % 2: # First wall is y
            d_new = torch.roll(d_new, 1, dims=-1)
        polygon = Polygon.from_Rd(self.R, torch.cat([self.d[..., :2], d_new], dim=-1))
        return polygon

    @autobatch
    def split(self, thr: float, max_num_walls: Optional[int] = None) -> 'Polygon':
        '''Split walls.

        Args:
            thr: Maximum wall width.
            max_num_walls: Maximum number of walls after splitting.
        Returns:
            polygon: Split polygon.
        '''
        d_wall = self.d[..., 2:][:self.num_walls]
        wall_widths = torch.abs(torch.roll(d_wall, -1) - torch.roll(d_wall, 1))
        d = deque([(x, w) for x, w in zip(self.d[..., 2:].cpu(), wall_widths.cpu())])

        while max_num_walls is None or len(d) < max_num_walls:
            idx = np.argmax([x[1] for x in d])
            width = d[idx][1]
            if width < thr:
                break
            d[idx] = (d[idx][0], width / 2.0)
            idx_prev = (idx - 1) % len(d)
            idx_next = (idx + 1) % len(d)
            d.insert(idx + 1, ((d[idx_prev][0] + d[idx_next][0]) / 2.0, 0.0))
            d.insert(idx + 2, (d[idx][0], width / 2.0))

        d_new = torch.tensor([x[0] for x in d], dtype=self.dtype, device=self.device)
        return Polygon.from_Rd(self.R, torch.cat([self.d[..., :2], d_new], dim=-1))

    @autobatch
    def make_valid(self) -> 'Polygon':
        '''Make the polygon valid by removing self-intersections.

        Returns:
            polygon: Fixed polygon.
        '''
        poly = shapely.Polygon(self.poly2d.detach().cpu().numpy())

        poly = shapely.make_valid(poly)
        while poly.geom_type in ('GeometryCollection', 'MultiPolygon'):
            idx = np.argmax([p.area for p in poly.geoms])
            poly = poly.geoms[idx]
        assert poly.geom_type == 'Polygon', f'Expected Polygon, got {poly.geom_type}'

        # In rare cases Shapely will return a polygon that has more than two
        # points along an edge, with equal x or y values. Run simplification
        # to remove the intermediate points.
        poly = shapely.simplify(poly, 0.0)

        xy = np.stack(poly.exterior.xy, axis=-1)[:-1]
        if area(xy) < 0:
            xy = xy[::-1].copy()
        return Polygon.from_poly2d(self.R, xy, self.d[..., 0], self.d[..., 1])
