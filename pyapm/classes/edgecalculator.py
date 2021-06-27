from numpy import absolute, arctan, logical_and, square, sqrt, log
from numpy import divide, multiply, sign, pi, where, arctan2
from numpy.matlib import matrix
from numpy.linalg import norm

fourPi = 4*pi
eightPi = 2*fourPi
piby2 = pi/2

class EdgeCalculator():
    x: matrix = None
    ya: matrix = None
    yb: matrix = None
    z: matrix = None
    _absx: matrix = None
    _absz: matrix = None
    _chkx0: matrix = None
    _chkz0: matrix = None
    _yainf: matrix = None
    _ybinf: matrix = None
    _chkp0: matrix = None
    _sgnx: matrix = None
    _sgnz: matrix = None
    _logx: matrix = None
    _logz: matrix = None
    _pb2ya: matrix = None
    _pb2yb: matrix = None
    _abszox: matrix = None
    _xz: matrix = None
    _x2: matrix = None
    _z2: matrix = None
    _Ra2: matrix = None
    _Rb2: matrix = None
    _Ra: matrix = None
    _Rb: matrix = None
    _ra: matrix = None
    _rb: matrix = None
    _p2: matrix = None
    _p: matrix = None
    _logp: matrix = None
    _yora: matrix = None
    _yorb: matrix = None
    _tha: matrix = None
    _thb: matrix = None
    _sgnzth: matrix = None
    _Ja: matrix = None
    _Jb: matrix = None
    _sgnzJ: matrix = None
    _Ka: matrix = None
    _Kb: matrix = None
    _K: matrix = None
    _ln2xo2: matrix = None
    _Qa: matrix = None
    _Qb: matrix = None
    _Q: matrix = None
    _La: matrix = None
    _Lb: matrix = None
    _Na: matrix = None
    _Nb: matrix = None
    _N: matrix = None
    _Ma: matrix = None
    _Mb: matrix = None
    _M: matrix = None
    _sgnzth: matrix = None
    _sintha: matrix = None
    _sinthb: matrix = None
    _costha: matrix = None
    _costhb: matrix = None
    _sinLa: matrix = None
    _sinLb: matrix = None
    _sinL: matrix = None
    _cosLa: matrix = None
    _cosLb: matrix = None
    _cosL: matrix = None
    _phido: matrix = None
    _phidx: matrix = None
    _phidy: matrix = None
    _phidxx: matrix = None
    _phidxy: matrix = None
    _phidyy: matrix = None
    _phiso: matrix = None
    _phisx: matrix = None
    _phisy: matrix = None
    def __init__(self, x: matrix, ya: matrix, yb: matrix, z: matrix, tol: float=1e-12):
        self.x = x
        self.ya = ya
        self.yb = yb
        self.z = z
        self.set_tolerance(tol=tol)
    def set_tolerance(self, tol: float=1e-12):
        self.x = where(self.absx < tol, 0.0, self.x)
        self.ya = where(absolute(self.ya) < tol, 0.0, self.ya)
        self.yb = where(absolute(self.yb) < tol, 0.0, self.yb)
        self.z = where(self.absz < tol, 0.0, self.z)
        self.reset()
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    @property
    def absx(self):
        if self._absx is None:
            self._absx = absolute(self.x)
        return self._absx
    @property
    def absz(self):
        if self._absz is None:
            self._absz = absolute(self.z)
        return self._absz
    @property
    def chkx0(self):
        if self._chkx0 is None:
            self._chkx0 = self.x == 0.0
        return self._chkx0
    @property
    def chkz0(self):
        if self._chkz0 is None:
            self._chkz0 = self.z == 0.0
        return self._chkz0
    @property
    def yainf(self):
        if self._yainf is None:
            self._yainf = self.ya == float('-inf')
        return self._yainf
    @property
    def ybinf(self):
        if self._ybinf is None:
            self._ybinf = self.yb == float('inf')
        return self._ybinf
    @property
    def sgnx(self):
        if self._sgnx is None:
            self._sgnx = sign(self.x)
        return self._sgnx
    @property
    def sgnz(self):
        if self._sgnz is None:
            self._sgnz = sign(self.z)
            self._sgnz = where(self.chkz0, -1.0, self._sgnz)
        return self._sgnz
    @property
    def logx(self):
        if self._logx is None:
            self._logx = log(self.absx)
            self._logx = where(self.chkx0, 0.0, self._logx)
        return self._logx
    @property
    def logz(self):
        if self._logz is None:
            self._logz = log(self.absz)
            self._logz = where(self.chkz0, 0.0, self._logz)
        return self._logz
    @property
    def x2(self):
        if self._x2 is None:
            self._x2 = square(self.x)
        return self._x2
    @property
    def xz(self):
        if self._xz is None:
            self._xz = multiply(self.x, self.z)
        return self._xz
    @property
    def z2(self):
        if self._z2 is None:
            self._z2 = square(self.z)
        return self._z2
    @property
    def Ra2(self):
        if self._Ra2 is None:
            self._Ra2 = self.x2 + square(self.ya)
        return self._Ra2
    @property
    def Rb2(self):
        if self._Rb2 is None:
            self._Rb2 = self.x2 + square(self.yb)
        return self._Rb2
    @property
    def Ra(self):
        if self._Ra is None:
            self._Ra = sqrt(self.Ra2)
        return self._Ra
    @property
    def Rb(self):
        if self._Rb is None:
            self._Rb = sqrt(self.Rb2)
        return self._Rb
    @property
    def ra(self):
        if self._ra is None:
            self._ra = sqrt(self.Ra2 + self.z2)
        return self._ra
    @property
    def rb(self):
        if self._rb is None:
            self._rb = sqrt(self.Rb2 + self.z2)
        return self._rb
    @property
    def p2(self):
        if self._p2 is None:
            self._p2 = self.x2 + self.z2
        return self._p2
    @property
    def p(self):
        if self._p is None:
            self._p = sqrt(self.p2)
        return self._p
    @property
    def chkp0(self):
        if self._chkp0 is None:
            self._chkp0 = logical_and(self.chkx0, self.chkz0)
        return self._chkp0
    @property
    def logp(self):
        if self._logp is None:
            self._logp = log(self.p)
            self._logp = where(self.chkp0, 0.0, self._logp)
        return self._logp
    @property
    def pb2ya(self):
        if self._pb2ya is None:
            self._pb2ya = sign(self.ya)*piby2
        return self._pb2ya
    @property
    def pb2yb(self):
        if self._pb2yb is None:
            self._pb2yb = sign(self.yb)*piby2
        return self._pb2yb
    @property
    def abszox(self):
        if self._abszox is None:
            self._abszox = divide(self.absz, self.x)
            self._abszox = where(self.chkx0, 0.0, self._abszox)
        return self._abszox
    @property
    def yora(self):
        if self._yora is None:
            self._yora = divide(self.ya, self.ra)
            self._yora = where(self.yainf, -1.0, self._yora)
        return self._yora
    @property
    def yorb(self):
        if self._yorb is None:
            self._yorb = divide(self.yb, self.rb)
            self._yorb = where(self.ybinf, 1.0, self._yorb)
        return self._yorb
    @property
    def Ja(self):
        if self._Ja is None:
            self._Ja = arctan(multiply(self.yora, self.abszox))
            self._Ja = where(self.chkx0, self.pb2ya, self._Ja)
        return self._Ja
    @property
    def Jb(self):
        if self._Jb is None:
            self._Jb = arctan(multiply(self.yorb, self.abszox))
            self._Jb = where(self.chkx0, self.pb2yb, self._Jb)
        return self._Jb
    @property
    def sgnzJ(self):
        if self._sgnzJ is None:
            self._sgnzJ = multiply(self.sgnz, self.Jb - self.Ja)
        return self._sgnzJ
    @property
    def tha(self):
        if self._tha is None:
            self._tha = arctan(divide(self.ya, self.x))
            self._tha = where(self.chkx0, self.pb2ya, self._tha)
        return self._tha
    @property
    def thb(self):
        if self._thb is None:
            self._thb = arctan(divide(self.yb, self.x))
            self._thb = where(self.chkx0, self.pb2yb, self._thb)
        return self._thb
    @property
    def sgnzth(self):
        if self._sgnzth is None:
            self._sgnzth = multiply(self.sgnz, self.thb - self.tha)
        return self._sgnzth
    @property
    def Qa(self):
        if self._Qa is None:
            rpyaop = divide(self.ra + self.ya, self.p)
            self._Qa = log(rpyaop)
            self._Qa = where(rpyaop == 0.0, 0.0, self._Qa)
            self._Qa = where(self.chkp0, 0.0, self._Qa)
            self._Qa = where(self.yainf, self.logp, self._Qa)
        return self._Qa
    @property
    def Qb(self):
        if self._Qb is None:
            rpybop = divide(self.rb + self.yb, self.p)
            self._Qb = log(rpybop)
            self._Qb = where(rpybop == 0.0, 0.0, self._Qb)
            self._Qb = where(self.chkp0, 0.0, self._Qb)
            self._Qb = where(self.ybinf, -self.logp, self._Qb)
        return self._Qb
    @property
    def Q(self):
        if self._Q is None:
            self._Q = self.Qb - self.Qa
        return self._Q
    @property
    def La(self):
        if self._La is None:
            Rpraoz = divide(self.Ra + self.ra, self.absz)
            self._La = log(Rpraoz)
            self._La = where(self.chkz0, 0.0, self._La)
            self._La = where(self.yainf, 0.0, self._La)
        return self._La
    @property
    def Lb(self):
        if self._Lb is None:
            Rprboz = divide(self.Rb + self.rb, self.absz)
            self._Lb = log(Rprboz)
            self._Lb = where(self.chkz0, 0.0, self._Lb)
            self._Lb = where(self.ybinf, 0.0, self._Lb)
        return self._Lb
    @property
    def sintha(self):
        if self._sintha is None:
            self._sintha = divide(self.ya, self.Ra)
            self._sintha = where(self.yainf, -1.0, self._sintha)
            self._sintha = where(self.Ra == 0.0, 0.0, self._sintha)
        return self._sintha
    @property
    def sinthb(self):
        if self._sinthb is None:
            self._sinthb = divide(self.yb, self.Rb)
            self._sinthb = where(self.ybinf, 1.0, self._sinthb)
            self._sinthb = where(self.Rb == 0.0, 0.0, self._sinthb)
        return self._sinthb
    @property
    def costha(self):
        if self._costha is None:
            self._costha = divide(self.x, self.Ra)
            self._costha = where(self.yainf, 0.0, self._costha)
            self._costha = where(self.Ra == 0.0, 0.0, self._costha)
        return self._costha
    @property
    def costhb(self):
        if self._costhb is None:
            self._costhb = divide(self.x, self.Rb)
            self._costhb = where(self.ybinf, 0.0, self._costhb)
            self._costhb = where(self.Rb == 0.0, 0.0, self._costhb)
        return self._costhb
    @property
    def sinLa(self):
        if self._sinLa is None:
            self._sinLa = multiply(self.sintha, self.La)
        return self._sinLa
    @property
    def sinLb(self):
        if self._sinLb is None:
            self._sinLb = multiply(self.sinthb, self.Lb)
        return self._sinLb
    @property
    def sinL(self):
        if self._sinL is None:
            self._sinL = self.sinLb-self.sinLa
        return self._sinL
    @property
    def cosLa(self):
        if self._cosLa is None:
            self._cosLa = multiply(self.costha, self.La)
        return self._cosLa
    @property
    def cosLb(self):
        if self._cosLb is None:
            self._cosLb = multiply(self.costhb, self.Lb)
        return self._cosLb
    @property
    def cosL(self):
        if self._cosL is None:
            self._cosL = self.cosLb-self.cosLa
        return self._cosL
    @property
    def Na(self):
        if self._Na is None:
            sincostha = multiply(self.sintha, self.costha)
            self._Na = multiply(self.absz - self.ra, sincostha)
            self._Na = where(self.yainf, self.x, self._Na)
        return self._Na
    @property
    def Nb(self):
        if self._Nb is None:
            sincosthb = multiply(self.sinthb, self.costhb)
            self._Nb = multiply(self.absz - self.rb, sincosthb)
            self._Nb = where(self.ybinf, -self.x, self._Nb)
        return self._Nb
    @property
    def N(self):
        if self._N is None:
            self._N = self.Nb-self.Na
        return self._N
    @property
    def Ma(self):
        if self._Ma is None:
            zsin2tha = multiply(self.absz, square(self.sintha))
            rcos2tha = multiply(self.ra, square(self.costha))
            self._Ma = rcos2tha + zsin2tha
            self._Ma = where(self.yainf, self.absz, self._Ma)
        return self._Ma
    @property
    def Mb(self):
        if self._Mb is None:
            zsin2thb = multiply(self.absz, square(self.sinthb))
            rcos2thb = multiply(self.rb, square(self.costhb))
            self._Mb = rcos2thb + zsin2thb
            self._Mb = where(self.ybinf, self.absz, self._Mb)
        return self._Mb
    @property
    def M(self):
        if self._M is None:
            self._M = self.Mb-self.Ma
        return self._M
    @property
    def Ka(self):
        if self._Ka is None:
            self._Ka = multiply(self.sgnz, arctan(divide(self.Na, self.Ma)))
            self._Ka = where(self.Ra == 0.0, 0.0, self._Ka)
        return self._Ka
    @property
    def Kb(self):
        if self._Kb is None:
            self._Kb = multiply(self.sgnz, arctan(divide(self.Nb, self.Mb)))
            self._Kb = where(self.Rb == 0.0, 0.0, self._Kb)
        return self._Kb
    @property
    def K(self):
        if self._K is None:
            self._K = self.Kb-self.Ka
            self._K = where(self.chkp0, 0.0, self._K)
        return self._K
    @property
    def phido(self):
        if self._phido is None:
            # sgnJmth = self.sgnzJ - self.sgnzth
            # self._phido = sgnJmth/fourPi
            self._phido = self.K/fourPi
        return self._phido
    @property
    def phidx(self):
        if self._phidx is None:
            self._phidx = multiply(self.z, self.Q-self.sinL)/fourPi
        return self._phidx
    @property
    def phidy(self):
        if self._phidy is None:
            self._phidy = multiply(self.z, self.cosL)/fourPi
        return self._phidy
    @property
    def phidxx(self):
        if self._phidxx is None:
            self._phidxx = multiply(self.z, self.N)/fourPi
            self._phidxx -= multiply(self.z2, self.phido)
        return self._phidxx
    @property
    def phidxy(self):
        if self._phidxy is None:
            self._phidxy = multiply(self.z, self.M)/fourPi
        return self._phidxy
    @property
    def phidyy(self):
        if self._phidyy is None:
            self._phidyy = (multiply(-self.z, self.N) - multiply(self.xz, self.Q))/fourPi
            self._phidyy -= multiply(self.z2, self.phido)
        return self._phidyy
    @property
    def phiso(self):
        if self._phiso is None:
            self._phiso = multiply(-self.x, self.Q)/fourPi
            self._phiso -= multiply(self.z, self.phido)
        return self._phiso
    @property
    def phisx(self):
        if self._phisx is None:
            self._phisx = multiply(-self.x2, self.Q)/eightPi
            self._phisx -=multiply(self.z, self.phidx/2)
        return self._phisx
    @property
    def phisy(self):
        if self._phisy is None:
            self._phisy = multiply(-self.x, self.rb-self.ra)/eightPi
            self._phisy -= multiply(self.z, self.phidy/2)
        return self._phisy
