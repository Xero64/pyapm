from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from numpy.matlib import zeros
from pygeom.geom3d import Vector, Coordinate
from .boundedge import BoundEdge
from typing import List

tol = 1e-12

class Poly(object):
    grds: List[Vector] = None
    _num: int = None
    _pnto: Vector = None
    _edgs: List[BoundEdge] = None
    _nrm: Vector = None
    _area: float = None
    def __init__(self, grds: List[Vector]):
        self.grds = grds
    @property
    def num(self):
        if self._num is None:
            self._num = len(self.grds)
        return self._num
    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = sum(self.grds)/self.num
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
            self._nrm = (sum([edg.dirz for edg in self.edgs])/self.num).to_unit()
        return self._nrm
    @property
    def area(self):
        if self._area is None:
            self._area = 0.0
            for edg in self.edgs:
                self._area += edg.area
        return self._area
    def influence_coefficient(self, pnt: Vector, display=False):
        phiv = 0.0
        phis = 0.0
        velv = Vector(0.0, 0.0, 0.0)
        vels = Vector(0.0, 0.0, 0.0)
        for edg in self.edgs:
            ephiv, ephis, evelv, evels = edg.influence_coefficient(pnt, display=display)
            phiv += ephiv
            phis += ephis
            velv += evelv
            vels += evels
        return phiv, phis, velv, vels
    def influence_coefficients(self, pnts: MatrixVector):
        phiv = zeros(pnts.shape, float)
        phis = zeros(pnts.shape, float)
        velv = zero_matrix_vector(pnts.shape, float)
        vels = zero_matrix_vector(pnts.shape, float)
        for edg in self.edgs:
            ephiv, ephis, evelv, evels = edg.influence_coefficients(pnts)
            phiv += ephiv
            phis += ephis
            velv += evelv
            vels += evels
        return phiv, phis, velv, vels
    def doublet_influence_coefficients(self, pnts: MatrixVector):
        phiv = zeros(pnts.shape, float)
        velv = zero_matrix_vector(pnts.shape, float)
        for edg in self.edgs:
            ephiv, evelv = edg.doublet_influence_coefficients(pnts)
            phiv += ephiv
            velv += evelv
        return phiv, velv
    def doublet_velocity_potentials(self, pnts: MatrixVector):
        phiv = zeros(pnts.shape, float)
        for edg in self.edgs:
            phiv += edg.doublet_velocity_potentials(pnts)
        return phiv
    def velocity_potentials(self, pnts: MatrixVector):
        phiv = zeros(pnts.shape, float)
        phis = zeros(pnts.shape, float)
        for edg in self.edgs:
            ephiv, ephis = edg.velocity_potentials(pnts)
            phiv += ephiv
            phis += ephis
        return phiv, phis
