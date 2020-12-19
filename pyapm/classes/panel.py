from .poly import Poly
from .grid import Grid
from .horseshoe import HorseShoe
from typing import List
from pygeom.geom3d import Vector, Coordinate, ihat, khat
from math import sqrt

oor2 = 1/sqrt(2.0)

class Panel(Poly):
    pid: int = None
    gids: List[int] = None
    ind: int = None
    _crd: Coordinate = None
    _hsvs: List[HorseShoe] = None
    def __init__(self, pid: int, gids: List[int]):
        self.pid = pid
        self.gids = gids
    def set_grids(self, grds: List[Grid]):
        super(Panel, self).__init__(grds)
    def set_index(self, ind: int):
        self.ind = ind
    @property
    def crd(self) -> Coordinate:
        if self._crd is None:
            dirz = self.nrm
            vecy = dirz**ihat
            magy = vecy.return_magnitude()
            if magy < oor2:
                vecy = dirz**khat
            diry = vecy.to_unit()
            dirx = (diry**dirz).to_unit()
            pntc = self.pnto.to_point()
            self._crd = Coordinate(pntc, dirx, diry, dirz)
        return self._crd
    @property
    def hsvs(self):
        if self._hsvs is None:
            self._hsvs = []
            for b in range(self.num):
                a = b+1
                if a == self.num:
                    a = 0
                if self.grds[a].te and self.grds[b].te:
                    self._hsvs.append(HorseShoe(self.grds[a], self.grds[b], ihat, self.ind))
        return self._hsvs
