from typing import TYPE_CHECKING

from numpy import (absolute, arctan, divide, log, logical_and, logical_not,
                   multiply, ones, pi, zeros)
from numpy.linalg import inv
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


tol = 1e-12
piby2 = pi/2
twoPi = 2*pi
fourPi = 4*pi

class DirichletPoly():
    grds: list[Vector] = None
    _num: int = None
    _pnto: Vector = None
    _grdr: Vector = None
    _vecab: Vector = None
    _vecaxb: Vector = None
    _sumaxb: Vector = None
    _nrm: Vector = None
    _area: float = None
    _dirxab: Vector = None
    _diryab: Vector = None
    _dirzab: Vector = None
    _baryinv: list['NDArray'] = None

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
            self._pnto = sum(self.grds, start=Vector(0.0, 0.0, 0.0))/self.num
        return self._pnto

    @property
    def grdr(self):
        if self._grdr is None:
            self._grdr = Vector.zeros((1, self.num))
            for i in range(self.num):
                self._grdr[0, i] = self.grds[i] - self.pnto
        return self._grdr

    def mach_grids(self, betx: float=1.0, bety: float=1.0, betz: float=1.0):
        grdm = self.grdr.copy()
        grdm.x = grdm.x/betx
        grdm.y = grdm.y/bety
        grdm.z = grdm.z/betz
        return grdm

    def edge_cross(self, grds: Vector):
        vecaxb = Vector.zeros((1, self.num))
        for i in range(self.num):
            veca = grds[0, i-1]
            vecb = grds[0, i]
            vecaxb[0, i] = veca.cross(vecb)
        return vecaxb

    def edge_vector(self, grds: Vector):
        vecab = Vector.zeros((1, self.num))
        for i in range(self.num):
            veca = grds[0, i-1]
            vecb = grds[0, i]
            vecab[0, i] = vecb - veca
        return vecab

    @property
    def vecab(self):
        if self._vecab is None:
            self._vecab = self.edge_vector(self.grdr)
        return self._vecab

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

    @property
    def dirxab(self):
        if self._dirxab is None:
            self._dirxab = self.vecab.to_unit()
        return self._dirxab

    @property
    def diryab(self):
        if self._diryab is None:
            self._diryab = self.dirzab.cross(self.dirxab)
        return self._diryab

    @property
    def dirzab(self):
        if self._dirzab is None:
            self._dirzab = self.vecaxb.to_unit()
        return self._dirzab

    @property
    def baryinv(self):
        if self._baryinv is None:
            self._baryinv = []
            for i in range(self.num):
                dirx = self.dirxab[0, i]
                diry = self.diryab[0, i]
                dirz = self.dirzab[0, i]
                grda = self.grdr[0, i-1]
                grdb = self.grdr[0, i]
                grdal = Vector(grda.dot(dirx), grda.dot(diry), grda.dot(dirz))
                grdbl = Vector(grdb.dot(dirx), grdb.dot(diry), grdb.dot(dirz))
                amat = zeros((3, 3))
                amat[0, :] = 1.0
                amat[1, 1] = grdal.x
                amat[2, 1] = grdal.y
                amat[1, 2] = grdbl.x
                amat[2, 2] = grdbl.y
                self._baryinv.append(inv(amat))
        return self._baryinv

    def relative_mach(self, pnts: Vector, pnt: Vector,
                      betx: float=1.0, bety: float=1.0, betz: float=1.0):
        vecs = pnts-pnt
        vecs.x = vecs.x/betx
        vecs.y = vecs.y/bety
        vecs.z = vecs.z/betz
        return vecs

    def influence_coefficients(self, pnts: Vector, incvel: bool=True,
                               betx: float=1.0, bety: float=1.0, betz: float=1.0,
                               checktol: bool=False):
        grdm = self.mach_grids(betx=betx, bety=bety, betz=betz)
        vecab = self.edge_vector(grdm)
        vecaxb = self.edge_cross(grdm)
        dirxab = vecab.to_unit()
        dirzab = vecaxb.to_unit()
        diryab = dirzab.cross(dirxab)
        nrm = vecaxb.sum().to_unit()
        rgcs = self.relative_mach(pnts, self.pnto, betx=betx, bety=bety, betz=betz)
        locz = rgcs.dot(nrm)
        sgnz = ones(locz.shape)
        sgnz[locz <= 0.0] = -1.0
        vecgcs = []
        for i in range(self.num):
            vecgcs.append(self.relative_mach(pnts, self.grds[i], betx=betx,
                                             bety=bety, betz=betz))
        phid = zeros(pnts.shape)
        phis = zeros(pnts.shape)
        if incvel:
            veld = Vector.zeros(pnts.shape)
            vels = Vector.zeros(pnts.shape)
        for i in range(self.num):
            # Edge Length
            dab = vecab[0, i].return_magnitude()
            # Local Coordinate System
            dirx = dirxab[0, i]
            diry = diryab[0, i]
            dirz = dirzab[0, i]
            # Vector A in Local Coordinate System
            veca = vecgcs[i-1]
            alcs = Vector(veca.dot(dirx), veca.dot(diry), veca.dot(dirz))
            if checktol:
                alcs.x[absolute(alcs.x) < tol] = 0.0
                alcs.y[absolute(alcs.y) < tol] = 0.0
                alcs.z[absolute(alcs.z) < tol] = 0.0
            # Vector A Doublet Velocity Potentials
            phida, amag = phi_doublet_array(alcs, sgnz)
            # Vector B in Local Coordinate System
            vecb = vecgcs[i]
            blcs = Vector(vecb.dot(dirx), vecb.dot(diry), vecb.dot(dirz))
            if checktol:
                blcs.x[absolute(blcs.x) < tol] = 0.0
                blcs.y[absolute(blcs.y) < tol] = 0.0
                blcs.z[absolute(blcs.z) < tol] = 0.0
            # Vector B Doublet Velocity Potentials
            phidb, bmag = phi_doublet_array(blcs, sgnz)
            # Edge Doublet Velocity Potentials
            phidi = phida - phidb
            # Edge Source Velocity Potentials
            phisi, Qab = phi_source_array(amag, bmag, dab, alcs, phidi)
            # Add Edge Velocity Potentials
            phid += phidi
            phis += phisi
            # Calculate Edge Velocities
            if incvel:
                # Velocities in Local Coordinate System
                veldi = vel_doublet_array(alcs, amag, blcs, bmag)
                velsi = vel_source_array(Qab, alcs, phidi)
                # Transform to Global Coordinate System and Add
                dirxi = Vector(dirx.x, diry.x, dirz.x)
                diryi = Vector(dirx.y, diry.y, dirz.y)
                dirzi = Vector(dirx.z, diry.z, dirz.z)
                veld += Vector(veldi.dot(dirxi), veldi.dot(diryi), veldi.dot(dirzi))
                vels += Vector(velsi.dot(dirxi), velsi.dot(diryi), velsi.dot(dirzi))
        phid = phid/fourPi
        phis = phis/fourPi
        if incvel:
            veld = veld/fourPi
            vels = vels/fourPi
            output = phid, phis, veld, vels
        else:
            output = phid, phis
        return output

    def velocity_potentials(self, pnts: Vector,
                            betx: float=1.0, bety: float=1.0, betz: float=1.0):
        phi = self.influence_coefficients(pnts, incvel=False,
                                          betx=betx, bety=bety, betz=betz)
        return phi[0], phi[1]

def phi_doublet_array(vecs: Vector, sgnz: 'NDArray') -> tuple['NDArray', 'NDArray']:
    mags = vecs.return_magnitude()
    chkm = mags < tol
    chky = absolute(vecs.y) < tol
    vecs.y[chky] = 0.0
    ms = zeros(mags.shape)
    divide(vecs.x, vecs.y, where=logical_not(chky), out=ms)
    ths = arctan(ms)
    ths[chky] = piby2
    ts = zeros(mags.shape)
    divide(vecs.z, mags, where=logical_not(chkm), out=ts)
    gs = multiply(ms, ts)
    Js = arctan(gs)
    Js[chky] = piby2
    phids = Js - multiply(sgnz, ths)
    return phids, mags

def phi_source_array(am: 'NDArray', bm: 'NDArray', dab: 'NDArray',
                     rl: Vector, phid: 'NDArray') -> tuple['NDArray', 'NDArray']:
    numrab = am + bm + dab
    denrab = am + bm - dab
    Pab = ones(numrab.shape)
    chkd = absolute(denrab) > tol
    divide(numrab, denrab, where=chkd, out=Pab)
    Qab = log(Pab)
    tmps = multiply(rl.y, Qab)
    phis = -multiply(rl.z, phid) - tmps
    return phis, Qab

def vel_doublet_array(av: Vector, am: 'NDArray',
                      bv: Vector, bm: 'NDArray') -> Vector:
    adb = av.dot(bv)
    abm = multiply(am, bm)
    dm = multiply(abm, abm+adb)
    axb = av.cross(bv)
    axbm = axb.return_magnitude()
    chki = (axbm == 0.0)
    chki = logical_and(axbm >= -tol, axbm <= tol)
    chkd = absolute(dm) < tol
    fac = zeros(axbm.shape)
    divide(am+bm, dm, where=logical_not(chkd), out=fac)
    velvl = axb*fac
    velvl.x[chki] = 0.0
    velvl.y[chki] = 0.0
    velvl.z[chki] = 0.0
    return velvl

def vel_source_array(Qab: 'NDArray', rl: Vector, phid: 'NDArray') -> Vector:
    velsl = Vector.zeros(Qab.shape)
    velsl.y = -Qab
    faco = ones(Qab.shape)
    faco[rl.z != 0.0] = -1.0
    velsl.z = multiply(faco, phid)
    return velsl
