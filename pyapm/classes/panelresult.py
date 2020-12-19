from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.matrix3d import elementwise_dot_product, elementwise_multiply, elementwise_cross_product
from numpy.matlib import matrix, ones
from math import cos, sin, radians
from numpy.matlib import sqrt, square, multiply

class PanelResult(object):
    name: str = None
    sys = None
    rho: float = None
    mach: float = None
    speed: float = None
    alpha: float = None
    beta: float = None
    pbo2V: float = None
    qco2V: float = None
    rbo2V: float = None
    _acs: Coordinate = None
    _vfs: Vector = None
    _qfs: float = None
    _ons: matrix = None
    _sig: matrix = None
    _mu: matrix = None
    _avm: matrix = None
    _vg: MatrixVector = None
    _vl: MatrixVector = None
    _vt: matrix = None
    _cp: matrix = None
    _phi: matrix = None
    _nrmfrc: matrix = None
    _frc: MatrixVector = None
    _mom: MatrixVector = None
    def __init__(self, name: str, sys):
        self.name = name
        self.sys = sys
        self.initialise()
    def initialise(self):
        self.rho = 1.0
        self.mach = 0.0
        self.speed = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        self.pbo2V = 0.0
        self.qco2V = 0.0
        self.rbo2V = 0.0
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    def set_density(self, rho: float=None):
        if rho is not None:
            self.rho = rho
        self.reset()
    def set_state(self, mach: float=None, speed: float=None,
                  alpha: float=None, beta: float=None,
                  pbo2V: float=None, qco2V: float=None, rbo2V: float=None):
        if mach is not None:
            self.mach = mach
        if speed is not None:
            self.speed = speed
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if pbo2V is not None:
            self.pbo2V = pbo2V
        if qco2V is not None:
            self.qco2V = qco2V
        if rbo2V is not None:
            self.rbo2V = rbo2V
        self.reset()
    @property
    def acs(self):
        if self._acs is None:
            pnt = self.sys.rref
            cosal, sinal = trig_angle(self.alpha)
            cosbt, sinbt = trig_angle(self.beta)
            dirx = Vector(cosbt*cosal, -sinbt, cosbt*sinal)
            diry = Vector(sinbt*cosal, cosbt, sinbt*sinal)
            dirz = Vector(-sinal, 0.0, cosal)
            self._acs = Coordinate(pnt, dirx, diry, dirz)
        return self._acs
    @property
    def vfs(self):
        if self._vfs is None:
            if self.alpha is None:
                self.alpha = 0.0
            if self.beta is None:
                self.beta = 0.0
            if self.speed is None:
                self.speed = 1.0
            self._vfs = self.acs.dirx*self.speed
        return self._vfs
    @property
    def qfs(self):
        if self._qfs is None:
            self._qfs = self.rho*self.speed**2/2
        return self._qfs
    @property
    def ons(self):
        if self._ons is None:
            self._ons = ones((self.sys.numpnl, 1), dtype=float)
        return self._ons
    @property
    def sig(self):
        if self._sig is None:
            self._sig = self.sys.sig*self.vfs
        return self._sig
    @property
    def mu(self):
        if self._mu is None:
            self._mu = self.sys.mu*self.vfs
        return self._mu
    @property
    def vg(self):
        if self._vg is None:
            self._vg = self.ons*self.vfs + self.sys.avs*self.sig + self.sys.avm*self.mu
        return self._vg
    @property
    def vl(self):
        if self._vl is None:
            self._vl = zero_matrix_vector(self.vg.shape, dtype=float)
            for pnl in self.sys.pnls.values():
                self._vl[pnl.ind, 0] = pnl.crd.vector_to_local(self.vg[pnl.ind, 0])
        return self._vl
    @property
    def vt(self):
        if self._vt is None:
            self._vt = sqrt(square(self.vl.x) + square(self.vl.y))
        return self._vt
    @property
    def cp(self):
        if self._cp is None:
            self._cp = self.ons - square(self.vt)/self.speed**2
        return self._cp
    @property
    def phi(self):
        if self._phi is None:
            self._phi = self.sys.apm*self.mu + self.sys.aps*self.sig
        return self._phi
    @property
    def nrmfrc(self):
        if self._nrmfrc is None:
            self._nrmfrc = -multiply(self.cp, self.sys.pnla)
        return self._nrmfrc
    @property
    def frc(self):
        if self._frc is None:
            self._frc = self.qfs*elementwise_multiply(self.sys.nrms, self.nrmfrc)
        return self._frc
    @property
    def mom(self):
        if self._mom is None:
            self._mom = elementwise_cross_product(self.sys.pntr, self.frc)
        return self._mom
    def __repr__(self):
        return f'<PanelResult: {self.name}>'
    def _repr_markdown_(self):
        return self.__str__()

def trig_angle(angle: float):
    '''Calculates cos(angle) and sin(angle) with angle in degrees.'''
    angrad = radians(angle)
    cosang = cos(angrad)
    sinang = sin(angrad)
    return cosang, sinang
