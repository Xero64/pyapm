from typing import TYPE_CHECKING

from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from .panel import Panel


class Grid(Vector):
    gid: int = None
    te: bool = None
    ind: int = None
    pnls: set['Panel'] = None

    def __init__(self, gid: int, x: float, y: float, z: float, te: bool=False):
        self.gid = gid
        self.te = te
        super().__init__(x, y, z)
        self.pnls = set()

    def set_index(self, ind: int):
        self.ind = ind

    def __str__(self):
        outstr = '{:}: <{:}, {:}, {:}>'.format(self.gid, self.x, self.y, self.z)
        if self.te:
            outstr += ', Trailing Edge: True'
        return outstr

    def __format__(self, format_spec):
        frmstr = '{:}: <{:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'}>'
        outstr = frmstr.format(self.gid, self.x, self.y, self.z)
        if self.te:
            outstr += ', Trailing Edge: True'
        return outstr
