from typing import TYPE_CHECKING

from numpy import (absolute, arctan, divide, log, logical_and, multiply, ones,
                   pi)
from pygeom.geom3d import Coordinate, Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


tol = 1e-12
piby2 = pi/2
fourPi = 4*pi

class BoundEdge():
    pnto: Vector = None
    grda: Vector = None
    grdb: Vector = None
    _vecab: Vector = None
    _lenab: Vector = None
    _veca: Vector = None
    _vecb: Vector = None
    _vecaxb: Vector = None
    _area: float = None
    _dirx: Vector = None
    _diry: Vector = None
    _dirz: Vector = None
    _pntc: Vector = None
    _crd: Coordinate = None
    _pntol: Vector = None
    _grdal: Vector = None
    _grdbl: Vector = None

    def __init__(self, pnto: Vector, grda: Vector, grdb: Vector):
        self.pnto = pnto
        self.grda = grda
        self.grdb = grdb

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
    def veca(self):
        if self._veca is None:
            self._veca = self.pnto - self.grda
        return self._veca

    @property
    def vecb(self):
        if self._vecb is None:
            self._vecb = self.pnto - self.grdb
        return self._vecb

    @property
    def vecaxb(self):
        if self._vecaxb is None:
            self._vecaxb = self.veca.cross(self.vecb)
        return self._vecaxb

    @property
    def area(self):
        if self._area is None:
            self._area = self._vecaxb.return_magnitude()/2
        return self._area

    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = self.vecab/self.lenab
        return self._dirx

    @property
    def dirz(self):
        if self._dirz is None:
            self._dirz = self.vecaxb.to_unit()
        return self._dirz

    @property
    def diry(self):
        if self._diry is None:
            self._diry = self.dirz.cross(self.dirx).to_unit()
        return self._diry

    @property
    def pntc(self):
        if self._pntc is None:
            lenx = self.veca.dot(self.dirx)
            self._pntc = self.grda + lenx*self.dirx
        return self._pntc

    @property
    def crd(self):
        if self._crd is None:
            self._crd = Coordinate(self.pntc, self.dirx, self.diry)
        return self._crd

    @property
    def pntol(self):
        if self._pntol is None:
            vec = self.pnto-self.pntc
            self._pntol = Vector(vec.dot(self.dirx), vec.dot(self.diry),
                                 vec.dot(self.dirz))
        return self._pntol

    @property
    def grdal(self):
        if self._grdal is None:
            vec = self.grda-self.pntc
            self._grdal = Vector(vec.dot(self.dirx), vec.dot(self.diry),
                                 vec.dot(self.dirz))
        return self._grdal

    @property
    def grdbl(self):
        if self._grdbl is None:
            vec = self.grdb-self.pntc
            self._grdbl = Vector(vec.dot(self.dirx), vec.dot(self.diry),
                                 vec.dot(self.dirz))
        return self._grdbl

    @property
    def te(self):
        if self.grda.te and self.grdb.te:
            return True
        else:
            return False

    def points_to_local(self, pnts: Vector, betx: float=1.0):
        vecs = pnts-self.pntc
        if betx != 1.0:
            vecs.x = vecs.x/betx
        return Vector(vecs.dot(self.dirx), vecs.dot(self.diry),
                            vecs.dot(self.dirz))

    def vectors_to_global(self, vecs: Vector):
        dirx = Vector(self.dirx.x, self.diry.x, self.dirz.x)
        diry = Vector(self.dirx.y, self.diry.y, self.dirz.y)
        dirz = Vector(self.dirx.z, self.diry.z, self.dirz.z)
        return Vector(vecs.dot(dirx), vecs.dot(diry), vecs.dot(dirz))

    def doublet_velocity_potentials(self, pnts: Vector, extraout: bool=False,
                                    sgnz: 'NDArray'=None, factor: bool=True,
                                    betx: float=1.0):
        rls = self.points_to_local(pnts, betx=betx)
        absx = absolute(rls.x)
        rls.x[absx < tol] = 0.0
        absy = absolute(rls.y)
        rls.y[absy < tol] = 0.0
        absz = absolute(rls.z)
        rls.z[absz < tol] = 0.0
        if sgnz is None:
            sgnz = ones(rls.shape, float)
            sgnz[rls.z <= 0.0] = -1.0
        avs = rls-self.grdal
        phida, ams = phi_doublet_array(avs, rls, sgnz)
        bvs = rls-self.grdbl
        phidb, bms = phi_doublet_array(bvs, rls, sgnz)
        phids = phida-phidb
        if factor:
            phids = phids/fourPi
        if extraout:
            return phids, rls, avs, ams, bvs, bms
        else:
            return phids

    def doublet_influence_coefficients(self, pnts: Vector,
                                       sgnz: 'NDArray'=None, factor: bool=True,
                                       betx: float=1.0):
        phid, _, av, am, bv, bm = self.doublet_velocity_potentials(pnts, extraout=True,
                                                                   sgnz=sgnz,
                                                                   factor=False,
                                                                   betx=betx)
        veldl = vel_doublet_array(av, am, bv, bm)
        veld = self.vectors_to_global(veldl)
        if factor:
            phid, veld = phid/fourPi, veld/fourPi
        return phid, veld

    def velocity_potentials(self, pnts: Vector,
                            sgnz: 'NDArray'=None, factor: bool=True, betx: float=1.0):
        phid, rl, _, am, _, bm = self.doublet_velocity_potentials(pnts, extraout=True,
                                                                  sgnz=sgnz,
                                                                  factor=False,
                                                                  betx=betx)
        phis, _ = phi_source_array(am, bm, self.lenab, rl, phid)
        if factor:
            phid, phis = phid/fourPi, phis/fourPi
        return phid, phis

    def influence_coefficients(self, pnts: Vector,
                               sgnz: 'NDArray'=None, factor: bool=True, betx: float=1.0):
        phid, rl, av, am, bv, bm = self.doublet_velocity_potentials(pnts, extraout=True,
                                                                    sgnz=sgnz,
                                                                    factor=False,
                                                                    betx=betx)
        phis, Qab = phi_source_array(am, bm, self.lenab, rl, phid)
        velsl = vel_source_array(Qab, rl, phid)
        vels = self.vectors_to_global(velsl)
        veldl = vel_doublet_array(av, am, bv, bm)
        veld = self.vectors_to_global(veldl)
        if factor:
            phid, phis, veld, vels = phid/fourPi, phis/fourPi, veld/fourPi, vels/fourPi
        return phid, phis, veld, vels

    def __str__(self):
        outstr = ''
        outstr += f'pnto = {self.pnto}\n'
        outstr += f'grda = {self.grda}\n'
        outstr += f'grdb = {self.grdb}\n'
        outstr += f'crd.pnt = {self.crd.pnt}\n'
        outstr += f'crd.dirx = {self.crd.dirx}\n'
        outstr += f'crd.diry = {self.crd.diry}\n'
        outstr += f'crd.dirz = {self.crd.dirz}\n'
        outstr += f'pntol = {self.pntol}\n'
        outstr += f'grdal = {self.grdal}\n'
        outstr += f'grdbl = {self.grdbl}\n'
        return outstr

def phi_doublet_array(vecs: Vector, rls: Vector, sgnz: 'NDArray'):
    mags = vecs.return_magnitude()
    ms = divide(vecs.x, rls.y)
    ths = arctan(ms)
    ths[rls.y == 0.0] = piby2
    gs = multiply(ms, divide(rls.z, mags))
    Js = arctan(gs)
    Js[rls.y == 0.0] = piby2
    phids = Js - multiply(sgnz, ths)
    return phids, mags

def phi_source_array(am, bm, dab, rl: Vector, phid):
    numrab = am+bm+dab
    denrab = am+bm-dab
    Pab = divide(numrab, denrab)
    Pab[denrab == 0.0] = 1.0
    Qab = log(Pab)
    tmps = multiply(rl.y, Qab)
    phis = -multiply(rl.z, phid) - tmps
    return phis, Qab

def vel_doublet_array(av: Vector, am, bv: Vector, bm):
    adb = av.dot(bv)
    abm = multiply(am, bm)
    dm = multiply(abm, abm+adb)
    axb = av.cross(bv)
    axbm = axb.return_magnitude()
    chki = (axbm == 0.0)
    chki = logical_and(axbm >= -tol, axbm <= tol)
    velvl = axb*divide(am+bm, dm)
    velvl.x[chki] = 0.0
    velvl.y[chki] = 0.0
    velvl.z[chki] = 0.0
    return velvl

def vel_source_array(Qab, rl, phid):
    velsl = Vector.zeros(Qab.shape)
    velsl.y = -Qab
    # velsl.z = -phid
    faco = ones(Qab.shape)
    faco[rl.z != 0.0] = -1.0
    velsl.z = multiply(faco, phid)
    return velsl
