from math import pi
from numpy.matlib import matrix, ones, absolute, divide, multiply, arctan, log, logical_and
from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.matrix3d import elementwise_dot_product, elementwise_cross_product
from pygeom.matrix3d import elementwise_multiply

tol = 1e-12
piby2 = pi/2
fourPi = 4*pi

class BoundEdge(object):
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
            self._vecaxb = self.veca**self.vecb
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
            self._diry = (self.dirz**self.dirx).to_unit()
        return self._diry
    @property
    def pntc(self):
        if self._pntc is None:
            lenx = self.veca*self.dirx
            self._pntc = self.grda + lenx*self.dirx
        return self._pntc
    @property
    def crd(self):
        if self._crd is None:
            self._crd = Coordinate(self.pntc, self.dirx, self.diry, self.dirz)
        return self._crd
    @property
    def pntol(self):
        if self._pntol is None:
            vec = self.pnto-self.pntc
            self._pntol = Vector(vec*self.dirx, vec*self.diry, vec*self.dirz)
        return self._pntol
    @property
    def grdal(self):
        if self._grdal is None:
            vec = self.grda-self.pntc
            self._grdal = Vector(vec*self.dirx, vec*self.diry, vec*self.dirz)
        return self._grdal
    @property
    def grdbl(self):
        if self._grdbl is None:
            vec = self.grdb-self.pntc
            self._grdbl = Vector(vec*self.dirx, vec*self.diry, vec*self.dirz)
        return self._grdbl
    @property
    def te(self):
        if self.grda.te and self.grdb.te:
            return True
        else:
            return False
    def points_to_local(self, pnts: MatrixVector, betx: float=1.0):
        vecs = pnts-self.pntc
        if betx != 1.0:
            vecs.x = vecs.x/betx
        return MatrixVector(vecs*self.dirx, vecs*self.diry, vecs*self.dirz)
    def vectors_to_global(self, vecs: MatrixVector):
        dirx = Vector(self.dirx.x, self.diry.x, self.dirz.x)
        diry = Vector(self.dirx.y, self.diry.y, self.dirz.y)
        dirz = Vector(self.dirx.z, self.diry.z, self.dirz.z)
        return MatrixVector(vecs*dirx, vecs*diry, vecs*dirz)
    def doublet_velocity_potentials(self, pnts: MatrixVector, extraout: bool=False,
                                    sgnz: matrix=None, factor: bool=True,
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
        phida, ams = phi_doublet_matrix(avs, rls, sgnz)
        bvs = rls-self.grdbl
        phidb, bms = phi_doublet_matrix(bvs, rls, sgnz)
        phids = phida-phidb
        if factor:
            phids = phids/fourPi
        if extraout:
            return phids, rls, avs, ams, bvs, bms
        else:
            return phids
    def doublet_influence_coefficients(self, pnts: MatrixVector,
                                       sgnz: matrix=None, factor: bool=True, betx: float=1.0):
        phid, _, av, am, bv, bm = self.doublet_velocity_potentials(pnts, extraout=True,
                                                                   sgnz=sgnz, factor=False,
                                                                   betx=betx)
        veldl = vel_doublet_matrix(av, am, bv, bm)
        veld = self.vectors_to_global(veldl)
        if factor:
            phid, veld = phid/fourPi, veld/fourPi
        return phid, veld
    def velocity_potentials(self, pnts: MatrixVector,
                            sgnz: matrix=None, factor: bool=True, betx: float=1.0):
        phid, rl, _, am, _, bm = self.doublet_velocity_potentials(pnts, extraout=True,
                                                                  sgnz=sgnz, factor=False,
                                                                  betx=betx)
        phis, _ = phi_source_matrix(am, bm, self.lenab, rl, phid)
        if factor:
            phid, phis = phid/fourPi, phis/fourPi
        return phid, phis
    def influence_coefficients(self, pnts: MatrixVector,
                               sgnz: matrix=None, factor: bool=True, betx: float=1.0):
        phid, rl, av, am, bv, bm = self.doublet_velocity_potentials(pnts, extraout=True,
                                                                    sgnz=sgnz, factor=False,
                                                                    betx=betx)
        phis, Qab = phi_source_matrix(am, bm, self.lenab, rl, phid)
        velsl = vel_source_matrix(Qab, rl, phid)
        vels = self.vectors_to_global(velsl)
        veldl = vel_doublet_matrix(av, am, bv, bm)
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

def phi_doublet_matrix(vecs: MatrixVector, rls: MatrixVector, sgnz: matrix):
    mags = vecs.return_magnitude()
    ms = divide(vecs.x, rls.y)
    ths = arctan(ms)
    ths[rls.y == 0.0] = piby2
    gs = multiply(ms, divide(rls.z, mags))
    Js = arctan(gs)
    Js[rls.y == 0.0] = piby2
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
    # velsl.z = -phid
    faco = ones(Qab.shape, dtype=float)
    faco[rl.z != 0.0] = -1.0
    velsl.z = multiply(faco, phid)
    return velsl
