from math import pi
from numpy.matlib import matrix, ones, zeros, absolute, divide, multiply, arctan, square
from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import MatrixVector, elementwise_divide
from .boundedge import phi_doublet_matrix

tol = 1e-12
piby2 = pi/2
fourPi = 4*pi
twoPi = 2*pi

class TrailingEdge(object):
    pnto: Vector = None
    grdo: Vector = None
    diro: Vector = None
    faco: Vector = None
    _veco: Vector = None
    _dirx: Vector = None
    _diry: Vector = None
    _dirz: Vector = None
    _pntc: Vector = None
    _crd: Coordinate = None
    _pntol: Vector = None
    _grdol: Vector = None
    def __init__(self, pnto: Vector, grdo: Vector, diro: Vector, faco: float):
        self.pnto = pnto
        self.grdo = grdo
        self.diro = diro.to_unit()
        self.faco = faco/abs(faco)
    @property
    def veco(self):
        if self._veco is None:
            self._veco = self.pnto - self.grdo
        return self._veco
    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = self.faco*self.diro
        return self._dirx
    @property
    def dirz(self):
        if self._dirz is None:
            self._dirz = (self.dirx**self.veco).to_unit()
        return self._dirz
    @property
    def diry(self):
        if self._diry is None:
            self._diry = (self.dirz**self.dirx).to_unit()
        return self._diry
    @property
    def pntc(self):
        if self._pntc is None:
            lenx = self.veco*self.dirx
            self._pntc = self.grdo + lenx*self.dirx
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
    def grdol(self):
        if self._grdol is None:
            vec = self.grdo-self.pntc
            self._grdol = Vector(vec*self.dirx, vec*self.diry, vec*self.dirz)
        return self._grdol
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
        ovs = rls-self.grdol
        phido, oms = phi_doublet_matrix(ovs, rls, sgnz)
        phidt = phi_trailing_doublet_matrix(rls, sgnz, self.faco)
        phid = phido*self.faco+phidt
        if factor:
            phid = phid/fourPi
        if extraout:
            output = phid, rls, ovs, oms
        else:
            output = phid
        return output
    def doublet_influence_coefficients(self, pnts: MatrixVector,
                                       sgnz: matrix=None, factor: bool=True,
                                       betx: float=1.0):
        phid, _, ov, om = self.doublet_velocity_potentials(pnts, extraout=True,
                                                           sgnz=sgnz, factor=False,
                                                           betx=betx)
        veldl = vel_doublet_matrix(ov, om, self.faco)
        veld = self.vectors_to_global(veldl)*self.faco
        if factor:
            phid, veld = phid/fourPi, veld/fourPi
        return phid, veld
    def trefftz_plane_velocities(self, pnts: MatrixVector):
        rls = self.points_to_local(pnts)
        rls.x = zeros(rls.shape, dtype=float)
        # ro = Vector(0.0, self.grdo.y, self.grdo.z)
        # o = rls-ro
        oxx = MatrixVector(rls.x, -rls.z, rls.y)
        om2 = square(rls.y) + square(rls.z)
        chkom2 = (absolute(om2) < tol)
        veldl = elementwise_divide(oxx, om2)*self.faco
        veldl.x[chkom2] = 0.0
        veldl.y[chkom2] = 0.0
        veldl.z[chkom2] = 0.0
        veld = self.vectors_to_global(veldl)*self.faco
        return veld/twoPi
    def __str__(self):
        outstr = ''
        outstr += f'pnto = {self.pnto}\n'
        outstr += f'grdo = {self.grdo}\n'
        outstr += f'crd.pnt = {self.crd.pnt}\n'
        outstr += f'crd.dirx = {self.crd.dirx}\n'
        outstr += f'crd.diry = {self.crd.diry}\n'
        outstr += f'crd.dirz = {self.crd.dirz}\n'
        outstr += f'pntol = {self.pntol}\n'
        outstr += f'grdol = {self.grdol}\n'
        return outstr

def vel_doublet_matrix(ov, om, faco):
    ov = faco*ov
    oxx = MatrixVector(zeros(ov.shape), -ov.z, ov.y)
    oxxm = oxx.return_magnitude()
    chko = (oxxm == 0.0)
    velol = elementwise_divide(oxx, multiply(om, om-ov.x))
    velol.x[chko] = 0.0
    velol.y[chko] = 0.0
    velol.z[chko] = 0.0
    return velol

def phi_trailing_doublet_matrix(rls: MatrixVector, sgnz: matrix, faco: float):
    ths = zeros(rls.shape, dtype=float)
    ths[rls.y > 0.0] = piby2
    ths[rls.y == 0.0] = -piby2*faco
    ths[rls.y < 0.0] = -piby2
    gs = divide(rls.z, rls.y)
    Js = arctan(gs)
    Js[rls.y == 0.0] = -piby2*faco
    phids = Js - multiply(sgnz, ths)
    return phids
