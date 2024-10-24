from math import pi

from numpy import square
from pygeom.geom2d import Vector2D

twoPi = 2*pi


class HorseshoeVortex2D():
    grda: Vector2D = None
    grdb: Vector2D = None
    _vecab: Vector2D = None
    _nrm: Vector2D = None
    _pnt: Vector2D = None

    def __init__(self, grda: Vector2D, grdb: Vector2D):
        self.grda = grda
        self.grdb = grdb

    @property
    def vecab(self):
        if self._vecab is None:
            self._vecab = self.grdb-self.grda
        return self._vecab

    @property
    def nrm(self):
        if self._nrm is None:
            self._nrm = Vector2D(-self.vecab.y, self.vecab.x).to_unit()
        return self._nrm

    @property
    def pnt(self):
        if self._pnt is None:
            self._pnt = (self.grdb+self.grda)/2
        return self._pnt

    def induced_velocity(self, pnts: Vector2D):
        agcs = pnts-self.grda
        amag = agcs.return_magnitude()
        vela = Vector2D(-agcs.y, agcs.x)/square(amag)
        bgcs = pnts-self.grdb
        bmag = bgcs.return_magnitude()
        velb = Vector2D(-bgcs.y, bgcs.x)/square(bmag)
        vel = (vela-velb)/twoPi
        return vel
