from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.geom3d import Vector, Coordinate, ihat, khat
from numpy.matlib import zeros
from math import atan2, sqrt, pi, atan
from .boundedge import BoundEdge
from .trailingedge import TrailingEdge

tol = 1e-12
fourPi = 4*pi
piby2 = pi/2
piby4 = pi/4

class HorseShoe(object):
    grda: Vector = None
    grdb: Vector = None
    diro: Vector = None
    ind: int = None
    _vecab: Vector = None
    _lenab: Vector = None
    _pntc: Vector = None
    _pnto: Vector = None
    _bvab: BoundEdge = None
    _tva: TrailingEdge = None
    _tvb: TrailingEdge = None
    def __init__(self, grda: Vector, grdb: Vector, diro: Vector, ind: int=None):
        self.grda = grda
        self.grdb = grdb
        self.diro = diro.to_unit()
        self.ind = ind
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    @property
    def vecab(self):
        if self._vecab is None:
            self._vecab = self.grdb - self.grda
        return self._vecab
    @property
    def lenab(self):
        if self._lenab is None:
            self._lenab = self._vecab.return_magnitude()
        return self._lenab
    @property
    def pntc(self):
        if self._pntc is None:
            self._pntc = self.grda + self.vecab/2
        return self._pntc
    @property
    def pnto(self):
        if self._pnto is None:
            self._pnto = self.pntc + self.lenab*self.diro/2
        return self._pnto
    @property
    def bvab(self):
        if self._bvab is None:
            self._bvab = BoundEdge(self.pnto, self.grda, self.grdb)
        return self._bvab
    @property
    def tva(self):
        if self._tva is None:
            self._tva = TrailingEdge(self.pnto, self.grda, self.diro, -1.0)
        return self._tva
    @property
    def tvb(self):
        if self._tvb is None:
            self._tvb = TrailingEdge(self.pnto, self.grdb, self.diro, 1.0)
        return self._tvb
    def doublet_influence_coefficients(self, pnts: MatrixVector):
        phid = zeros(pnts.shape, float)
        veld = zero_matrix_vector(pnts.shape, float)
        phidab, veldab = self.bvab.doublet_influence_coefficients(pnts)
        phida, velda = self.tva.doublet_influence_coefficients(pnts)
        phidb, veldb = self.tvb.doublet_influence_coefficients(pnts)
        phid = phidab + phida + phidb
        veld = veldab + velda + veldb
        return phid, veld
    def doublet_velocity_potentials(self, pnts: MatrixVector):
        phid = zeros(pnts.shape, float)
        phidab = self.bvab.doublet_velocity_potentials(pnts)
        phida = self.tva.doublet_velocity_potentials(pnts)
        phidb = self.tvb.doublet_velocity_potentials(pnts)
        phid = phidab + phida + phidb
        return phid
