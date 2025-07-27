from numpy import ones, zeros
from pygeom.geom3d import Vector

from .boundedge import BoundEdge


class Poly():
    grds: list[Vector] = None
    _num: int = None
    _pnto: Vector = None
    _edgs: list[BoundEdge] = None
    _nrm: Vector = None
    _area: float = None

    def __init__(self, grds: list[Vector]):
        self.grds = grds

    @property
    def num(self):
        if self._num is None:
            self._num = len(self.grds)
        return self._num

    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = sum(self.grds, Vector.zeros())/self.num
        return self._pnto

    @property
    def edgs(self):
        if self._edgs is None:
            self._edgs = []
            for a, grda in enumerate(self.grds):
                b = a + 1
                if b == self.num:
                    b = 0
                grdb = self.grds[b]
                self._edgs.append(BoundEdge(self.pnto, grda, grdb))
        return self._edgs

    @property
    def nrm(self):
        if self._nrm is None:
            edgnrml = Vector.from_iter([edg.dirz for edg in self.edgs])
            self._nrm = (edgnrml.sum()/self.num).to_unit()
        return self._nrm

    @property
    def area(self):
        if self._area is None:
            self._area = 0.0
            for edg in self.edgs:
                self._area += edg.area
        return self._area

    def sign_local_z(self, pnts: Vector, betx: float=1.0):
        vecs = pnts-self.pnto
        nrm = self.nrm
        if betx != 1.0:
            vecs.x = vecs.x/betx
            nrm = Vector(self.nrm.x/betx, self.nrm.y, self.nrm.z)
        locz = vecs.dot(nrm)
        sgnz = ones(locz.shape)
        sgnz[locz <= 0.0] = -1.0
        return sgnz

    def influence_coefficients(self, pnts: Vector, betx: float=1.0):
        phiv = zeros(pnts.shape)
        phis = zeros(pnts.shape)
        velv = Vector.zeros(pnts.shape)
        vels = Vector.zeros(pnts.shape)
        sgnz = self.sign_local_z(pnts, betx=betx)
        for edg in self.edgs:
            ephiv, ephis, evelv, evels = edg.influence_coefficients(pnts, sgnz=sgnz,
                                                                    betx=betx)
            phiv += ephiv
            phis += ephis
            velv += evelv
            vels += evels
        return phiv, phis, velv, vels

    def doublet_influence_coefficients(self, pnts: Vector, betx: float=1.0):
        phiv = zeros(pnts.shape)
        velv = Vector.zeros(pnts.shape)
        sgnz = self.sign_local_z(pnts, betx=betx)
        for edg in self.edgs:
            ephiv, evelv = edg.doublet_influence_coefficients(pnts, sgnz=sgnz,
                                                              betx=betx)
            phiv += ephiv
            velv += evelv
        return phiv, velv

    def doublet_velocity_potentials(self, pnts: Vector, betx: float=1.0):
        phiv = zeros(pnts.shape)
        sgnz = self.sign_local_z(pnts, betx=betx)
        for edg in self.edgs:
            phiv += edg.doublet_velocity_potentials(pnts, sgnz=sgnz, betx=betx)
        return phiv

    def velocity_potentials(self, pnts: Vector, betx: float=1.0):
        phiv = zeros(pnts.shape)
        phis = zeros(pnts.shape)
        sgnz = self.sign_local_z(pnts, betx=betx)
        for edg in self.edgs:
            ephiv, ephis = edg.velocity_potentials(pnts, sgnz=sgnz, betx=betx)
            phiv += ephiv
            phis += ephis
        return phiv, phis
