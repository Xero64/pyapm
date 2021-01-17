from numpy.matlib import zeros, ones
from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.geom3d import Vector
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
    _width: float = None
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
    @property
    def width(self):
        if self._width is None:
            diry = (self.nrm**self.diro).to_unit()
            grday = self.grda*diry
            grdby = self.grdb*diry
            self._width = grdby - grday
        return self._width
    def sign_local_z(self, pnts: MatrixVector, betm: float=1.0):
        vecs = pnts-self.pnto
        nrm = self.nrm
        if betm != 1.0:
            vecs.x = vecs.x/betm
            nrm = Vector(self.nrm.x/betm, self.nrm.y, self.nrm.z)
        locz = vecs*nrm
        sgnz = ones(locz.shape, float)
        sgnz[locz <= 0.0] = -1.0
        return sgnz
    def doublet_influence_coefficients(self, pnts: MatrixVector, betm: float=1.0):
        phid = zeros(pnts.shape, dtype=float)
        veld = zero_matrix_vector(pnts.shape, dtype=float)
        sgnz = self.sign_local_z(pnts, betm=betm)
        phida, velda = self.tva.doublet_influence_coefficients(pnts, sgnz=sgnz, betm=betm)
        phidb, veldb = self.tvb.doublet_influence_coefficients(pnts, sgnz=sgnz, betm=betm)
        phidab, veldab = self.bvab.doublet_influence_coefficients(pnts, sgnz=sgnz, betm=betm)
        phid = phida + phidb + phidab
        veld = velda + veldb + veldab
        return phid, veld
    def doublet_velocity_potentials(self, pnts: MatrixVector, betm: float=1.0):
        phid = zeros(pnts.shape, dtype=float)
        sgnz = self.sign_local_z(pnts, betm=betm)
        phida = self.tva.doublet_velocity_potentials(pnts, sgnz=sgnz, betm=betm)
        phidb = self.tvb.doublet_velocity_potentials(pnts, sgnz=sgnz, betm=betm)
        phidab = self.bvab.doublet_velocity_potentials(pnts, sgnz=sgnz, betm=betm)
        phid = phida + phidb + phidab
        return phid
    def trefftz_plane_velocities(self, pnts: MatrixVector):
        velda = self.tva.trefftz_plane_velocities(pnts)
        veldb = self.tvb.trefftz_plane_velocities(pnts)
        veld = velda + veldb
        return veld
