from typing import TYPE_CHECKING

from numpy import zeros
from numpy.linalg import inv
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RigidBody():
    pnt: list[Vector] = None
    pnts: list[Vector] = None
    ks: list[Vector] = None
    gs: list[Vector] = None
    _numpnt: int = None
    _rrel: list[Vector] = None
    _amat: 'NDArray' = None
    _ainv: 'NDArray' = None

    def __init__(self, pnt: Vector, pnts: list[Vector], ks: list[Vector], gs: list[Vector]):
        self.pnt = pnt
        self.pnts = []
        self.ks = []
        self.gs = []
        for pnti, ki, gi in zip(pnts, ks, gs):
            self.pnts.append(pnti)
            self.ks.append(ki)
            self.gs.append(gi)

    @property
    def numpnt(self):
        if self._numpnt is None:
            self._numpnt = len(self.pnts)
        return self._numpnt

    @property
    def rrel(self):
        if self._rrel is None:
            self._rrel = [pnti - self.pnt for pnti in self.pnts]
        return self._rrel

    @property
    def amat(self):
        if self._amat is None:
            self._amat = zeros((6, 6))
            for i in range(self.numpnt):
                ri = self.rrel[i]
                ki = self.ks[i]
                gi = self.gs[i]
                rix2 = ri.x**2
                riy2 = ri.y**2
                riz2 = ri.z**2
                self._amat[0, 0] += ki.x
                self._amat[0, 1] += 0.0
                self._amat[0, 2] += 0.0
                self._amat[1, 0] += 0.0
                self._amat[1, 1] += ki.y
                self._amat[1, 2] += 0.0
                self._amat[2, 0] += 0.0
                self._amat[2, 1] += 0.0
                self._amat[2, 2] += ki.z
                self._amat[0, 3] += 0.0
                self._amat[0, 4] += ki.x * ri.z
                self._amat[0, 5] -= ki.x * ri.y
                self._amat[1, 3] -= ki.y * ri.z
                self._amat[1, 4] += 0.0
                self._amat[1, 5] += ki.y * ri.x
                self._amat[2, 3] += ki.z * ri.y
                self._amat[2, 4] -= ki.z * ri.x
                self._amat[2, 5] += 0.0
                self._amat[3, 0] += 0.0
                self._amat[3, 1] -= ki.y * ri.z
                self._amat[3, 2] += ki.z * ri.y
                self._amat[4, 0] += ki.x * ri.z
                self._amat[4, 1] += 0.0
                self._amat[4, 2] -= ki.z * ri.x
                self._amat[5, 0] -= ki.x * ri.y
                self._amat[5, 1] += ki.y * ri.x
                self._amat[5, 2] += 0.0
                self._amat[3, 3] += gi.x + riy2 * ki.z + riz2 * ki.y
                self._amat[3, 4] -= ri.x * ri.y * ki.z
                self._amat[3, 5] -= ri.z * ri.x * ki.y
                self._amat[4, 3] -= ri.x * ri.y * ki.z
                self._amat[4, 4] += gi.y + riz2 * ki.x + rix2 * ki.z
                self._amat[4, 5] -= ri.y * ri.z * ki.x
                self._amat[5, 3] -= ri.z * ri.x * ki.y
                self._amat[5, 4] -= ri.y * ri.z * ki.x
                self._amat[5, 5] += gi.z + rix2 * ki.y + riy2 * ki.x
        return self._amat

    @property
    def ainv(self):
        if self._ainv is None:
            if self.numpnt > 0:
                self._ainv = inv(self.amat)
            else:
                self._ainv = zeros((6, 6))
        return self._ainv

    def return_reactions(self, frc: Vector, mom: Vector):
        pmat = zeros((6, 1))
        pmat[0, 0] = frc.x
        pmat[1, 0] = frc.y
        pmat[2, 0] = frc.z
        pmat[3, 0] = mom.x
        pmat[4, 0] = mom.y
        pmat[5, 0] = mom.z
        dmat = self.ainv*pmat
        utl = Vector(dmat[0, 0], dmat[1, 0], dmat[2, 0])
        url = Vector(dmat[3, 0], dmat[4, 0], dmat[5, 0])
        frcs = []
        moms = []
        for ki, gi, ri in zip(self.ks, self.gs, self.rrel):
            ui = utl - ri.cross(url)
            frcs.append(-Vector(ki.x*ui.x, ki.y*ui.y, ki.z*ui.z))
            moms.append(-Vector(gi.x*url.x, gi.y*url.y, gi.z*url.z))
        return frcs, moms
