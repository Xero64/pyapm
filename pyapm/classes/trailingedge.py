from typing import TYPE_CHECKING

from numpy import (absolute, arctan, divide, multiply, ones, pi, reciprocal,
                   square, zeros)
from pygeom.geom3d import Coordinate, Vector

from .boundedge import phi_doublet_array

if TYPE_CHECKING:
    from numpy.typing import NDArray

tol = 1e-12
piby2 = pi/2
fourPi = 4*pi
twoPi = 2*pi


class TrailingEdge():
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
            self._dirz = self.dirx.cross(self.veco).to_unit()
        return self._dirz

    @property
    def diry(self):
        if self._diry is None:
            self._diry = self.dirz.cross(self.dirx).to_unit()
        return self._diry

    @property
    def pntc(self):
        if self._pntc is None:
            lenx = self.veco.dot(self.dirx)
            self._pntc = self.grdo + lenx*self.dirx
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
    def grdol(self):
        if self._grdol is None:
            vec = self.grdo-self.pntc
            self._grdol = Vector(vec.dot(self.dirx), vec.dot(self.diry),
                                 vec.dot(self.dirz))
        return self._grdol

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

    def doublet_velocity_potentials(self, pnts: Vector, extraout: bool = False,
                                    sgnz: 'NDArray' = None, factor: bool = True,
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
        phido, oms = phi_doublet_array(ovs, rls, sgnz)
        phidt = phi_trailing_doublet_array(rls, sgnz, self.faco)
        phid = phido*self.faco+phidt
        if factor:
            phid = phid/fourPi
        if extraout:
            output = phid, rls, ovs, oms
        else:
            output = phid
        return output

    def doublet_influence_coefficients(self, pnts: Vector,
                                       sgnz: 'NDArray' = None, factor: bool=True,
                                       betx: float=1.0) -> tuple['NDArray', Vector]:
        phid, _, ov, om = self.doublet_velocity_potentials(pnts, extraout=True,
                                                           sgnz=sgnz, factor=False,
                                                           betx=betx)
        veldl = vel_doublet_array(ov, om, self.faco)
        veld = self.vectors_to_global(veldl)*self.faco
        if factor:
            phid, veld = phid/fourPi, veld/fourPi
        return phid, veld

    def trefftz_plane_velocities(self, pnts: Vector) -> Vector:
        rls = self.points_to_local(pnts)
        rls.x = zeros(rls.shape)
        # ro = Vector(0.0, self.grdo.y, self.grdo.z)
        # o = rls-ro
        oxx = Vector(rls.x, -rls.z, rls.y)
        om2 = square(rls.y) + square(rls.z)
        chkom2 = (absolute(om2) < tol)
        veldl = oxx/om2*self.faco
        veldl.x[chkom2] = 0.0
        veldl.y[chkom2] = 0.0
        veldl.z[chkom2] = 0.0
        veld = self.vectors_to_global(veldl)*self.faco
        return veld/twoPi

    def __str__(self) -> str:
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

def vel_doublet_array(ov: Vector, om: 'NDArray', faco: float) -> Vector:
    ov = faco*ov
    oxx = Vector(zeros(ov.shape), -ov.z, ov.y)
    deno = om*(om - ov.x)
    recn = zeros(deno.shape)
    chkd = absolute(deno) > tol
    reciprocal(deno, out=recn, where=chkd)
    velol = oxx*recn
    return velol

def phi_trailing_doublet_array(rls: Vector, sgnz: 'NDArray', faco: float) -> 'NDArray':
    ths = zeros(rls.shape)
    ths[rls.y > 0.0] = piby2
    ths[rls.y == 0.0] = -piby2*faco
    ths[rls.y < 0.0] = -piby2
    gs = divide(rls.z, rls.y)
    Js = arctan(gs)
    Js[rls.y == 0.0] = -piby2*faco
    phids = Js - multiply(sgnz, ths)
    return phids
