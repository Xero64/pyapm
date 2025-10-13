from typing import TYPE_CHECKING

from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from .panel import Panel


class Grid(Vector):
    gid: int = None
    ind: int = None
    panels: set['Panel'] = None

    def __init__(self, gid: int, x: float, y: float, z: float) -> None:
        self.gid = gid
        super().__init__(x, y, z)
        self.panels = set()

    def __repr__(self) -> str:
        return f'Grid({self.gid}, {self.x}, {self.y}, {self.z})'

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, frm: str) -> str:
        return f'Grid({self.gid:d}, {self.x:{frm}}, {self.y:{frm}}, {self.z:{frm}})'
