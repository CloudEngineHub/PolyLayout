from typing import Union

from .wrappers import Pose, Camera  # noqa
from .cuboid import Cuboid
from .polygon import Polygon

Layout = Union[Cuboid, Polygon]
