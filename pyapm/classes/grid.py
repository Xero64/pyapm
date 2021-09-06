from typing import List
from pygeom.geom3d import Vector

class Grid(Vector):
    gid: int = None
    te: bool = None
    ze: bool = None
    ind: int = None
    pnls: List[object] = None
    def __init__(self, gid: int, x: float, y: float, z: float) -> None:
        self.gid = gid
        self.te = False
        self.ze = False
        self.pnls = []
        super().__init__(x, y, z)
    def set_index(self, ind: int) -> None:
        self.ind = ind
    def __repr__(self) -> str:
        return f'<Grid {self.gid:d}: {self.x:}, {self.y:}, {self.z:}>'
    def __str__(self) -> str:
        outstr = '{:}: <{:}, {:}, {:}>'.format(self.gid, self.x, self.y, self.z)
        if self.te:
            outstr += ', Trailing Edge: True'
        return outstr
    def __format__(self, format_spec) -> str:
        frmstr = '{:}: <{:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'}>'
        outstr = frmstr.format(self.gid, self.x, self.y, self.z)
        if self.te:
            outstr += ', Trailing Edge: True'
        return outstr

class GridNormal(Vector):
    ind: int = None
    def __init__(self, grd, x, y, z) -> None:
        self.grd = grd
        super().__init__(x, y, z)
    def set_index(self, ind):
        self.ind = ind
