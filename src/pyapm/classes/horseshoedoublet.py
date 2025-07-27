from typing import TYPE_CHECKING

from numpy import (absolute, arctan, divide, logical_not, multiply, ones, pi,
                   reciprocal, square, zeros)
from pygeom.geom3d import Vector

from .dirichletpoly import phi_doublet_array, vel_doublet_array

if TYPE_CHECKING:
    from numpy.typing import NDArray


tol = 1e-12
piby2 = pi/2
fourPi = 4*pi
twoPi = 2*pi


class HorseshoeDoublet():
    grda: Vector = None
    grdb: Vector = None
    ind: int = None
    _vecab: Vector = None
    _lenab: Vector = None
    _pntc: Vector = None
    _pnto: Vector = None
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
    def nrm(self):
        if self._nrm is None:
            self._nrm = self.vecab.cross(self.diro).to_unit()
        return self._nrm

    @property
    def width(self):
        if self._width is None:
            diry = self.nrm.cross(self.diro).to_unit()
            grday = self.grda.dot(diry)
            grdby = self.grdb.dot(diry)
            self._width = grdby - grday
        return self._width

    def relative_mach(self, pnts: Vector, pnt: Vector,
                      betx: float=1.0, bety: float=1.0, betz: float=1.0):
        vecs = pnts-pnt
        vecs.x = vecs.x/betx
        vecs.y = vecs.y/bety
        vecs.z = vecs.z/betz
        return vecs

    def doublet_influence_coefficients(self, pnts: Vector, incvel: bool=True,
                                       betx: float=1.0, bety: float=1.0,
                                       betz: float=1.0, checktol: bool=False):
        vecab = Vector(self.vecab.x/betx, self.vecab.y/bety, self.vecab.z/betz)
        dirxab = vecab.to_unit()
        dirzab = Vector(self.nrm.x, self.nrm.y, self.nrm.z)
        diryab = dirzab.cross(dirxab)
        agcs = self.relative_mach(pnts, self.grda, betx=betx, bety=bety, betz=betz)
        bgcs = self.relative_mach(pnts, self.grdb, betx=betx, bety=bety, betz=betz)
        locz = agcs.dot(self.nrm)
        sgnz = ones(locz.shape)
        sgnz[locz <= 0.0] = -1.0
        phid = zeros(pnts.shape)
        if incvel:
            veld = Vector.zeros(pnts.shape)
        # Vector A in Local Coordinate System
        alcs = Vector(agcs.dot(dirxab), agcs.dot(diryab), agcs.dot(dirzab))
        if checktol:
            alcs.x[absolute(alcs.x) < tol] = 0.0
            alcs.y[absolute(alcs.y) < tol] = 0.0
            alcs.z[absolute(alcs.z) < tol] = 0.0
        # Vector A Doublet Velocity Potentials
        phida, amag = phi_doublet_array(alcs, sgnz)
        # Vector B in Local Coordinate System
        blcs = Vector(bgcs.dot(dirxab), bgcs.dot(diryab), bgcs.dot(dirzab))
        if checktol:
            blcs.x[absolute(blcs.x) < tol] = 0.0
            blcs.y[absolute(blcs.y) < tol] = 0.0
            blcs.z[absolute(blcs.z) < tol] = 0.0
        # Vector B Doublet Velocity Potentials
        phidb, bmag = phi_doublet_array(blcs, sgnz)
        # Edge Doublet Velocity Potentials
        phidi = phida - phidb
        # Add Edge Velocity Potentials
        phid += phidi
        if incvel:
            # Bound Edge Velocities in Local Coordinate System
            veldi = vel_doublet_array(alcs, amag, blcs, bmag)
            # Transform to Global Coordinate System and Add
            dirxi = Vector(dirxab.x, diryab.x, dirzab.x)
            diryi = Vector(dirxab.y, diryab.y, dirzab.y)
            dirzi = Vector(dirxab.z, diryab.z, dirzab.z)
            veld += Vector(veldi.dot(dirxi), veldi.dot(diryi), veldi.dot(dirzi))
        # Trailing Edge A Coordinate Transformation
        dirxa = self.diro
        dirza = self.nrm
        dirya = -dirza.cross(dirxa)
        alcs = Vector(agcs.dot(dirxa), agcs.dot(dirya), agcs.dot(dirza))
        # Trailing Edge A Velocity Potential
        phida, amag = phi_doublet_array(alcs, sgnz)
        phidt = phi_trailing_doublet_array(alcs, sgnz)
        phidi = phida + phidt
        # Add Trailing Edge A Velocity Potentials
        phid += phidi
        # Trailing Edge B Coordinate Transformation
        if incvel:
            # Trailing Edge A Velocities in Local Coordinate System
            veldi = vel_trailing_doublet_array(alcs, amag, 1.0)
            # Transform to Global Coordinate System and Add
            dirxi = Vector(dirxa.x, dirya.x, dirza.x)
            diryi = Vector(dirxa.y, dirya.y, dirza.y)
            dirzi = Vector(dirxa.z, dirya.z, dirza.z)
            veld += Vector(veldi.dot(dirxi), veldi.dot(diryi), veldi.dot(dirzi))
        # Trailing Edge B Coordinate Transformation
        dirxb = self.diro
        dirzb = self.nrm
        diryb = dirzb.cross(dirxb)
        blcs = Vector(bgcs.dot(dirxb), bgcs.dot(diryb), bgcs.dot(dirzb))
        # Trailing Edge B Velocity Potential
        phidb, bmag = phi_doublet_array(blcs, sgnz)
        phidt = phi_trailing_doublet_array(blcs, sgnz)
        phidi = phidb + phidt
        # Add Trailing Edge B Velocity Potentials
        phid += phidi
        if incvel:
            # Trailing Edge B Velocities in Local Coordinate System
            veldi = vel_trailing_doublet_array(blcs, bmag, 1.0)
            # Transform to Global Coordinate System and Add
            dirxi = Vector(dirxb.x, diryb.x, dirzb.x)
            diryi = Vector(dirxb.y, diryb.y, dirzb.y)
            dirzi = Vector(dirxb.z, diryb.z, dirzb.z)
            veld += Vector(veldi.dot(dirxi), veldi.dot(diryi), veldi.dot(dirzi))
        # Factors and Outputs
        phid = phid/fourPi
        if incvel:
            veld = veld/fourPi
            output = phid, veld
        else:
            output = phid
        return output

    def doublet_velocity_potentials(self, pnts: Vector,
                                    betx: float=1.0, bety: float=1.0, betz: float=1.0):
        phi = self.doublet_influence_coefficients(pnts, incvel=False,
                                                  betx=betx, bety=bety, betz=betz)
        return phi

    def trefftz_plane_velocities(self, pnts: Vector,
                                 betx: float=1.0, bety: float=1.0, betz: float=1.0):
        # Trailing Vortex A
        agcs = self.relative_mach(pnts, self.grda, betx=betx, bety=bety, betz=betz)
        dirxa = -self.diro
        dirza = self.nrm
        dirya = dirza.cross(dirxa)
        alcs = Vector(agcs.dot(dirxa), agcs.dot(dirya), agcs.dot(dirza))
        alcs.x = zeros(alcs.shape)
        axx = Vector(alcs.x, -alcs.z, alcs.y)
        am2 = square(alcs.y) + square(alcs.z)
        chkam2 = absolute(am2) < tol
        am2r = zeros(pnts.shape)
        reciprocal(am2, where=logical_not(chkam2), out=am2r)
        faca = -1.0
        veldl = axx*am2r*faca
        veldl.x[chkam2] = 0.0
        veldl.y[chkam2] = 0.0
        veldl.z[chkam2] = 0.0
        dirxi = Vector(dirxa.x, dirya.x, dirza.x)
        diryi = Vector(dirxa.y, dirya.y, dirza.y)
        dirzi = Vector(dirxa.z, dirya.z, dirza.z)
        velda = Vector(veldl.dot(dirxi), veldl.dot(diryi), veldl.dot(dirzi))*faca
        # Trailing Vortex B
        bgcs = self.relative_mach(pnts, self.grdb, betx=betx, bety=bety, betz=betz)
        dirxb = self.diro
        dirzb = self.nrm
        diryb = dirzb.cross(dirxb)
        blcs = Vector(bgcs.dot(dirxb), bgcs.dot(diryb), bgcs.dot(dirzb))
        blcs.x = zeros(blcs.shape)
        bxx = Vector(blcs.x, -blcs.z, blcs.y)
        bm2 = square(blcs.y) + square(blcs.z)
        chkbm2 = absolute(bm2) < tol
        bm2r = zeros(pnts.shape)
        reciprocal(bm2, where=logical_not(chkbm2), out=bm2r)
        facb = 1.0
        veldl = bxx*bm2r*facb
        veldl.x[chkbm2] = 0.0
        veldl.y[chkbm2] = 0.0
        veldl.z[chkbm2] = 0.0
        dirxi = Vector(dirxb.x, diryb.x, dirzb.x)
        diryi = Vector(dirxb.y, diryb.y, dirzb.y)
        dirzi = Vector(dirxb.z, diryb.z, dirzb.z)
        veldb = Vector(veldl.dot(dirxi), veldl.dot(diryi), veldl.dot(dirzi))*facb
        # Add Together
        veld = velda + veldb
        return veld/twoPi

def vel_trailing_doublet_array(ov, om, faco):
    ov: Vector = faco*ov
    oxx = Vector(zeros(ov.shape), -ov.z, ov.y)
    oxxm = oxx.return_magnitude()
    chko = absolute(oxxm) < tol
    den = multiply(om, om-ov.x)
    chkd = absolute(den) < tol
    denr = zeros(ov.shape)
    reciprocal(den, where=logical_not(chkd), out=denr)
    velol = oxx*denr
    velol.x[chko] = 0.0
    velol.y[chko] = 0.0
    velol.z[chko] = 0.0
    return velol

def phi_trailing_doublet_array(rls: Vector, sgnz: 'NDArray'):
    ths = zeros(rls.shape)
    chky = absolute(rls.y) < tol
    ths[rls.y > 0.0] = piby2
    ths[rls.y < 0.0] = -piby2
    ths[chky] = -piby2
    gs = zeros(rls.shape)
    divide(rls.z, rls.y, where=logical_not(chky), out=gs)
    Js = arctan(gs)
    Js[rls.y == 0.0] = -piby2
    phids = Js - multiply(sgnz, ths)
    return phids
