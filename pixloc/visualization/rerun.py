from typing import Optional

import rerun as rr
from rerun.datatypes import UVec3DArrayLike, Vec3DArrayLike
from rerun.datatypes.rgba32 import Rgba32ArrayLike, Rgba32Like

from ..pixlib.geometry import Camera, Cuboid, Layout, Polygon, Pose


def log_camera(
    name: str,
    camera: Camera,
    T_w2cam: Pose,
    image_plane_distance: Optional[float] = None,
) -> None:
    rr.log(
        name,
        rr.Transform3D(translation=T_w2cam.t, mat3x3=T_w2cam.R, from_parent=True),
    )
    rr.log(
        name,
        rr.Pinhole(
            focal_length=camera.f,
            principal_point=camera.c,
            width=camera.size[0],
            height=camera.size[1],
            image_plane_distance=image_plane_distance,
        ),
    )


def log_cuboid(name: str, cuboid: Cuboid, color: Optional[Rgba32Like] = None) -> None:
    rr.log(
        name,
        rr.Transform3D(translation=cuboid.t, mat3x3=cuboid.R, from_parent=True),
    )
    rr.log(name, rr.Boxes3D(sizes=cuboid.s, colors=color))


def log_polygon(name: str, polygon: Polygon, color: Optional[Rgba32Like] = None) -> None:
    rr.log(name, rr.LineStrips3D(polygon.corners[polygon.edges].tolist(), colors=color))


def log_layout(name: str, layout: Layout, color: Optional[Rgba32Like] = None) -> None:
    if isinstance(layout, Cuboid):
        log_cuboid(name, layout, color)
    elif isinstance(layout, Polygon):
        log_polygon(name, layout, color)


def log_image(name: str, image, opacity: Optional[float] = None) -> None:
    rr.log(name, rr.Image(image, opacity=opacity))


def log_points(
    name: str,
    points: Vec3DArrayLike,
    colors: Optional[Rgba32ArrayLike] = None,
    radii: Optional[float] = None,
) -> None:
    rr.log(name, rr.Points3D(points, colors=colors, radii=radii))


def log_mesh(
    name: str,
    verts: Vec3DArrayLike,
    faces: UVec3DArrayLike,
) -> None:
    rr.log(name, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces))
