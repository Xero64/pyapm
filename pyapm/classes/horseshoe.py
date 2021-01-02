from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.geom3d import Vector, Coordinate, ihat, khat
from numpy.matlib import zeros, ones
from .boundedge import BoundEdge
from .trailingedge import TrailingEdge

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
    _nrm: Vector = None
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
    @property
    def nrm(self):
        if self._nrm is None:
            self._nrm = self.bvab.dirz
        return self._nrm
    def sign_local_z(self, pnts: MatrixVector):
        locz = (pnts - self.pnto)*self.nrm
        sgnz = ones(locz.shape, float)
        sgnz[locz <= 0.0] = -1.0
        return sgnz
    def doublet_influence_coefficients(self, pnts: MatrixVector):
        phid = zeros(pnts.shape, float)
        veld = zero_matrix_vector(pnts.shape, float)
        sgnz = self.sign_local_z(pnts)
        phidab, veldab = self.bvab.doublet_influence_coefficients(pnts, sgnz=sgnz)
        phida, velda = self.tva.doublet_influence_coefficients(pnts, sgnz=sgnz)
        phidb, veldb = self.tvb.doublet_influence_coefficients(pnts, sgnz=sgnz)
        phid = phidab + phida + phidb
        veld = veldab + velda + veldb
        return phid, veld
    def doublet_velocity_potentials(self, pnts: MatrixVector):
        phid = zeros(pnts.shape, float)
        sgnz = self.sign_local_z(pnts)
        phidab = self.bvab.doublet_velocity_potentials(pnts, sgnz=sgnz)
        phida = self.tva.doublet_velocity_potentials(pnts, sgnz=sgnz)
        phidb = self.tvb.doublet_velocity_potentials(pnts, sgnz=sgnz)
        phid = phidab + phida + phidb
        return phid
