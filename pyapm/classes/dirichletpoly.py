from math import pi
from typing import List
from pygeom.geom3d import Vector
from pygeom.matrix3d import MatrixVector, zero_matrix_vector, elementwise_multiply
from pygeom.matrix3d import elementwise_cross_product, elementwise_dot_product
from numpy.matlib import matrix, zeros, ones, divide, arctan, multiply, logical_and, absolute, log

tol = 1e-12
piby2 = pi/2
fourPi = 4*pi

class DirichletPoly(object):
    grds: List[Vector] = None
    _num: int = None
    _pnto: Vector = None
    _grdr: MatrixVector = None
    _vecaxb: MatrixVector = None
    _sumaxb: Vector = None
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
    def grdr(self):
        if self._grdr is None:
            self._grdr = zero_matrix_vector((1, self.num), dtype=float)
            for i in range(self.num):
                self._grdr[0, i] = self.grds[i] - self.pnto
        return self._grdr
    def mach_grids(self, betx: float=1.0, bety: float=1.0, betz: float=1.0):
        grdm = self.grdr.copy()
        grdm.x = grdm.x/betx
        grdm.y = grdm.y/bety
        grdm.z = grdm.z/betz
        return grdm
    def edge_cross(self, grds: MatrixVector):
        vecaxb = zero_matrix_vector((1, self.num), dtype=float)
        for i in range(self.num):
            veca = grds[0, i-1]
            vecb = grds[0, i]
            vecaxb[0, i] = veca**vecb
        return vecaxb
    def edge_vector(self, grds: MatrixVector):
        vecab = zero_matrix_vector((1, self.num), dtype=float)
        for i in range(self.num):
            veca = grds[0, i-1]
            vecb = grds[0, i]
            vecab[0, i] = vecb-veca
        return vecab
    @property
    def vecaxb(self):
        if self._vecaxb is None:
            self._vecaxb = self.edge_cross(self.grdr)
        return self._vecaxb
    @property
    def sumaxb(self):
        if self._sumaxb is None:
            self._sumaxb = self.vecaxb.sum()
        return self._sumaxb
    @property
    def area(self):
        if self._area is None:
            self._area = self.sumaxb.return_magnitude()/2
        return self._area
    @property
    def nrm(self):
        if self._nrm is None:
            self._nrm = self.sumaxb.to_unit()
        return self._nrm
    def relative_mach(self, pnts: MatrixVector, pnt: Vector,
                      betx: float=1.0, bety: float=1.0, betz: float=1.0):
        vecs = pnts-pnt
        vecs.x = vecs.x/betx
        vecs.y = vecs.y/bety
        vecs.z = vecs.z/betz
        return vecs
    def influence_coefficients(self, pnts: MatrixVector, incvel: bool=True,
                               betx: float=1.0, bety: float=1.0, betz: float=1.0,
                               checktol: bool=False):
        grdm = self.mach_grids(betx=betx, bety=bety, betz=betz)
        vecab = self.edge_vector(grdm)
        vecaxb = self.edge_cross(grdm)
        dirxab = vecab.to_unit()
        dirzab = vecaxb.to_unit()
        diryab = elementwise_cross_product(dirzab, dirxab)
        nrm = vecaxb.sum().to_unit()
        rgcs = self.relative_mach(pnts, self.pnto, betx=betx, bety=bety, betz=betz)
        locz = rgcs*nrm
        sgnz = ones(locz.shape, dtype=float)
        sgnz[locz <= 0.0] = -1.0
        vecgcs = []
        for i in range(self.num):
            vecgcs.append(self.relative_mach(pnts, self.grds[i], betx=betx, bety=bety, betz=betz))
        phid = zeros(pnts.shape, dtype=float)
        phis = zeros(pnts.shape, dtype=float)
        if incvel:
            veld = zero_matrix_vector(pnts.shape, dtype=float)
            vels = zero_matrix_vector(pnts.shape, dtype=float)
        for i in range(self.num):
            # Edge Length
            dab = vecab[0, i].return_magnitude()
            # Local Coordinate System
            dirx = dirxab[0, i]
            diry = diryab[0, i]
            dirz = dirzab[0, i]
            # Vector A in Local Coordinate System
            veca = vecgcs[i-1]
            alcs = MatrixVector(veca*dirx, veca*diry, veca*dirz)
            if checktol:
                alcs.x[absolute(alcs.x) < tol] = 0.0
                alcs.y[absolute(alcs.y) < tol] = 0.0
                alcs.z[absolute(alcs.z) < tol] = 0.0
            # Vector A Doublet Velocity Potentials
            phida, amag = phi_doublet_matrix(alcs, sgnz)
            # Vector B in Local Coordinate System
            vecb = vecgcs[i]
            blcs = MatrixVector(vecb*dirx, vecb*diry, vecb*dirz)
            if checktol:
                blcs.x[absolute(blcs.x) < tol] = 0.0
                blcs.y[absolute(blcs.y) < tol] = 0.0
                blcs.z[absolute(blcs.z) < tol] = 0.0
            # Vector B Doublet Velocity Potentials
            phidb, bmag = phi_doublet_matrix(blcs, sgnz)
            # Edge Doublet Velocity Potentials
            phidi = phida - phidb
            # Edge Source Velocity Potentials
            phisi, Qab = phi_source_matrix(amag, bmag, dab, alcs, phidi)
            # Add Edge Velocity Potentials
            phid += phidi
            phis += phisi
            # Calculate Edge Velocities
            if incvel:
                # Velocities in Local Coordinate System
                veldi = vel_doublet_matrix(alcs, amag, blcs, bmag)
                velsi = vel_source_matrix(Qab, alcs, phidi)
                # Transform to Global Coordinate System and Add
                dirxi = Vector(dirx.x, diry.x, dirz.x)
                diryi = Vector(dirx.y, diry.y, dirz.y)
                dirzi = Vector(dirx.z, diry.z, dirz.z)
                veld += MatrixVector(veldi*dirxi, veldi*diryi, veldi*dirzi)
                vels += MatrixVector(velsi*dirxi, velsi*diryi, velsi*dirzi)
        phid = phid/fourPi
        phis = phis/fourPi
        if incvel:
            veld = veld/fourPi
            vels = vels/fourPi
            output = phid, phis, veld, vels
        else:
            output = phid, phis
        return output
    def velocity_potentials(self, pnts: MatrixVector,
                            betx: float=1.0, bety: float=1.0, betz: float=1.0):
        phi = self.influence_coefficients(pnts, incvel=False,
                                          betx=betx, bety=bety, betz=betz)
        return phi[0], phi[1]

def phi_doublet_matrix(vecs: MatrixVector, sgnz: matrix):
    mags = vecs.return_magnitude()
    ms = divide(vecs.x, vecs.y)
    ths = arctan(ms)
    ths[vecs.y == 0.0] = piby2
    gs = multiply(ms, divide(vecs.z, mags))
    Js = arctan(gs)
    Js[vecs.y == 0.0] = piby2
    phids = Js - multiply(sgnz, ths)
    return phids, mags

def phi_source_matrix(am, bm, dab, rl, phid):
    numrab = am+bm+dab
    denrab = am+bm-dab
    Pab = divide(numrab, denrab)
    Pab[denrab == 0.0] = 1.0
    Qab = log(Pab)
    tmps = multiply(rl.y, Qab)
    phis = -multiply(rl.z, phid) - tmps
    return phis, Qab

def vel_doublet_matrix(av, am, bv, bm):
    adb = elementwise_dot_product(av, bv)
    abm = multiply(am, bm)
    dm = multiply(abm, abm+adb)
    axb = elementwise_cross_product(av, bv)
    axbm = axb.return_magnitude()
    chki = (axbm == 0.0)
    chki = logical_and(axbm >= -tol, axbm <= tol)
    velvl = elementwise_multiply(axb, divide(am+bm, dm))
    velvl.x[chki] = 0.0
    velvl.y[chki] = 0.0
    velvl.z[chki] = 0.0
    return velvl

def vel_source_matrix(Qab, rl, phid):
    velsl = zero_matrix_vector(Qab.shape, dtype=float)
    velsl.y = -Qab
    faco = ones(Qab.shape, dtype=float)
    faco[rl.z != 0.0] = -1.0
    velsl.z = multiply(faco, phid)
    return velsl
