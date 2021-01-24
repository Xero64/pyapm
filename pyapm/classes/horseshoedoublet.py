from math import pi
from pygeom.geom3d import Vector
from pygeom.matrix3d import MatrixVector, elementwise_divide, zero_matrix_vector
from numpy.matlib import matrix, zeros, multiply, divide, arctan, ones, absolute
from .dirichletpoly import phi_doublet_matrix, vel_doublet_matrix

tol = 1e-12
piby2 = pi/2
fourPi = 4*pi

class HorseshoeDoublet(object):
    grda: Vector = None
    grdb: Vector = None
    ind: int = None
    _vecab: Vector = None
    _lenab: Vector = None
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
    def nrm(self):
        if self._nrm is None:
            self._nrm = (self.vecab**self.diro).to_unit()
        return self._nrm
    @property
    def width(self):
        if self._width is None:
            diry = (self.nrm**self.diro).to_unit()
            grday = self.grda*diry
            grdby = self.grdb*diry
            self._width = grdby - grday
        return self._width
    # def sign_local_z(self, pnts: MatrixVector, betm: float=1.0):
    #     vecs = pnts-self.pnto
    #     nrm = self.nrm
    #     if betm != 1.0:
    #         vecs.x = vecs.x/betm
    #         nrm = Vector(self.nrm.x/betm, self.nrm.y, self.nrm.z)
    #     locz = vecs*nrm
    #     sgnz = ones(locz.shape, float)
    #     sgnz[locz <= 0.0] = -1.0
    #     return sgnz
    # def doublet_influence_coefficients(self, pnts: MatrixVector, betm: float=1.0):
    #     phid = zeros(pnts.shape, dtype=float)
    #     veld = zero_matrix_vector(pnts.shape, dtype=float)
    #     sgnz = self.sign_local_z(pnts, betm=betm)
    #     phida, velda = self.tva.doublet_influence_coefficients(pnts, sgnz=sgnz, betm=betm)
    #     phidb, veldb = self.tvb.doublet_influence_coefficients(pnts, sgnz=sgnz, betm=betm)
    #     phidab, veldab = self.bvab.doublet_influence_coefficients(pnts, sgnz=sgnz, betm=betm)
    #     phid = phida + phidb + phidab
    #     veld = velda + veldb + veldab
    #     return phid, veld
    def relative_mach(self, pnts: MatrixVector, pnt: Vector,
                      betx: float=1.0, bety: float=1.0, betz: float=1.0):
        vecs = pnts-pnt
        vecs.x = vecs.x/betx
        vecs.y = vecs.y/bety
        vecs.z = vecs.z/betz
        return vecs
    def doublet_influence_coefficients(self, pnts: MatrixVector, incvel: bool=True,
                                       betx: float=1.0, bety: float=1.0, betz: float=1.0,
                                       checktol: bool=False):
        vecab = Vector(self.vecab.x/betx, self.vecab.y/bety, self.vecab.z/betz)
        dirxab = vecab.to_unit()
        dirzab = Vector(self.nrm.x, self.nrm.y, self.nrm.z)
        diryab = dirzab**dirxab
        agcs = self.relative_mach(pnts, self.grda, betx=betx, bety=bety, betz=betz)
        bgcs = self.relative_mach(pnts, self.grdb, betx=betx, bety=bety, betz=betz)
        locz = agcs*self.nrm
        sgnz = ones(locz.shape, dtype=float)
        sgnz[locz <= 0.0] = -1.0
        phid = zeros(pnts.shape, dtype=float)
        if incvel:
            veld = zero_matrix_vector(pnts.shape, dtype=float)
        # Vector A in Local Coordinate System
        alcs = MatrixVector(agcs*dirxab, agcs*diryab, agcs*dirzab)
        if checktol:
            alcs.x[absolute(alcs.x) < tol] = 0.0
            alcs.y[absolute(alcs.y) < tol] = 0.0
            alcs.z[absolute(alcs.z) < tol] = 0.0
        # Vector A Doublet Velocity Potentials
        phida, amag = phi_doublet_matrix(alcs, sgnz)
        # Vector B in Local Coordinate System
        blcs = MatrixVector(bgcs*dirxab, bgcs*diryab, bgcs*dirzab)
        if checktol:
            blcs.x[absolute(blcs.x) < tol] = 0.0
            blcs.y[absolute(blcs.y) < tol] = 0.0
            blcs.z[absolute(blcs.z) < tol] = 0.0
        # Vector B Doublet Velocity Potentials
        phidb, bmag = phi_doublet_matrix(blcs, sgnz)
        # Edge Doublet Velocity Potentials
        phidi = phida - phidb
        # Add Edge Velocity Potentials
        phid += phidi
        if incvel:
            # Bound Edge Velocities in Local Coordinate System
            veldi = vel_doublet_matrix(alcs, amag, blcs, bmag)
            # Transform to Global Coordinate System and Add
            dirxi = Vector(dirxab.x, diryab.x, dirzab.x)
            diryi = Vector(dirxab.y, diryab.y, dirzab.y)
            dirzi = Vector(dirxab.z, diryab.z, dirzab.z)
            veld += MatrixVector(veldi*dirxi, veldi*diryi, veldi*dirzi)
        # Trailing Edge A Coordinate Transformation
        dirxa = self.diro
        dirza = self.nrm
        dirya = -dirza**dirxa
        alcs = MatrixVector(agcs*dirxa, agcs*dirya, agcs*dirza)
        # Trailing Edge A Velocity Potential
        phida, amag = phi_doublet_matrix(alcs, sgnz)
        phidt = phi_trailing_doublet_matrix(alcs, sgnz, -1.0)
        phidi = phida + phidt
        # Add Trailing Edge A Velocity Potentials
        phid += phidi
        # Trailing Edge B Coordinate Transformation
        if incvel:
            # Trailing Edge A Velocities in Local Coordinate System
            veldi = vel_trailing_doublet_matrix(alcs, amag, 1.0)
            # Transform to Global Coordinate System and Add
            dirxi = Vector(dirxa.x, dirya.x, dirza.x)
            diryi = Vector(dirxa.y, dirya.y, dirza.y)
            dirzi = Vector(dirxa.z, dirya.z, dirza.z)
            veld += MatrixVector(veldi*dirxi, veldi*diryi, veldi*dirzi)
        # Trailing Edge B Coordinate Transformation
        dirxb = self.diro
        dirzb = self.nrm
        diryb = dirzb**dirxb
        blcs = MatrixVector(bgcs*dirxb, bgcs*diryb, bgcs*dirzb)
        # Trailing Edge B Velocity Potential
        phidb, bmag = phi_doublet_matrix(blcs, sgnz)
        phidt = phi_trailing_doublet_matrix(blcs, sgnz, 1.0)
        phidi = phidb + phidt
        # Add Trailing Edge B Velocity Potentials
        phid += phidi
        if incvel:
            # Trailing Edge B Velocities in Local Coordinate System
            veldi = vel_trailing_doublet_matrix(blcs, bmag, 1.0)
            # Transform to Global Coordinate System and Add
            dirxi = Vector(dirxb.x, diryb.x, dirzb.x)
            diryi = Vector(dirxb.y, diryb.y, dirzb.y)
            dirzi = Vector(dirxb.z, diryb.z, dirzb.z)
            veld += MatrixVector(veldi*dirxi, veldi*diryi, veldi*dirzi)
        # Factors and Outputs
        phid = phid/fourPi
        if incvel:
            veld = veld/fourPi
            output = phid, veld
        else:
            output = phid
        return output
    def velocity_potentials(self, pnts: MatrixVector,
                            betx: float=1.0, bety: float=1.0, betz: float=1.0):
        phi = self.doublet_influence_coefficients(pnts, incvel=False,
                                                  betx=betx, bety=bety, betz=betz)
        return phi

def vel_trailing_doublet_matrix(ov, om, faco):
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
