from typing import List
from .spacing import full_cosine_spacing, equal_spacing, linear_bias_left
from math import log, pi, copysign, atan

class NACA6Series(object):
    code: str = None
    cnum: int = None
    cspc: str = None
    _cdst: List[float] = None
    _xc: List[float] = None
    _yc: List[float] = None
    _dydx: List[float] = None
    _thc: List[float] = None
    _a: float = None
    _xx: float = None
    _q: float = None
    _cl: float = None
    _h: float = None
    _g: float = None
    _cff: float = None
    _cs: float = None
    def __init__(self, code: str, cnum: int=80):
        if code[0] != '6':
            print('Not a NACA 6 series code.')
            return None
        self.code = code
        self.update(cnum)
    def update(self, cnum: int, cspc: str='full-cosine'):
        self.cnum = cnum
        self.cspc = cspc
        self.reset()
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    @property
    def a(self):
        if self._a is None:
            self._a = float(self.code[1])/10
        return self._a
    @property
    def xx(self):
        if self._xx is None:
            self._xx = float(self.code[-2:])/100
        return self._xx
    @property
    def q(self):
        if self._q is None:
            self._q = float(self.code[-3])
        return self._q
    @property
    def cl(self):
        if self._cl is None:
            clstr = self.code[2:-3].lstrip('(').rstrip(')')
            self._cl = float(clstr)/10
        return self._cl
    @property
    def g(self):
        if self._g is None:
            self._g = (self.a**2*(0.5*log(self.a) - 0.25) + 0.25)/(self.a - 1)
        return self._g
    @property
    def h(self):
        if self._h is None:
            self._h = (0.25 - 0.5*log(1.0 - self.a))*(self.a - 1.0) + self.g
        return self._h
    @property
    def cff(self):
        if self._cff is None:
            self._cff = self.cl/(2*pi*(self.a+1))
        return self._cff
    @property
    def cs(self):
        if self._cs is None:
            self._cs = self.cl/(2*pi*(self.a + 1.0))
        return self._cs
    @property
    def cdst(self):
        if self._cdst is None:
            if self.cspc == 'full-cosine':
                self._cdst = full_cosine_spacing(self.cnum)
            elif self.cspc == 'equal':
                self._cdst = equal_spacing(self.cnum)
            else:
                return ValueError('Incorrect distribution on NACA6Series')
        return self._cdst
    @property
    def xc(self):
        if self._xc is None:
            self._xc = linear_bias_left(self.cdst, 0.2)
        return self._xc
    @property
    def yc(self):
        if self._yc is None:
            self._yc = []
            for xi in self.xc:
                if xi == 0.0:
                    self._yc.append(0.0)
                else:
                    self._yc.append(self.cl*((self.a - 1)*(self.g - self.h*xi - xi*log(xi)) - 0.5*(self.a - xi)**2*log(abs(self.a - xi)) + 0.25*(self.a - xi)**2 + 0.5*(xi - 1)**2*log(1 - xi) - 0.25*(xi - 1)**2)/(2*pi*(self.a - 1)*(self.a + 1)))
        return self._yc
    @property
    def dydx(self):
        if self._dydx is None:
            self._dydx = []
            for xi in self.xc:
                if xi == 0.0:
                    self._dydx.append(float('inf'))
                elif xi == self.a:
                    self._dydx.append(self.cs*(log(1 - self.a) - log(self.a) - self.h - 1))
                elif xi == 1.0:
                    self._dydx.append(self.cs*(copysign((self.a - xi)**2/2, self.a - xi) + ((xi-self.a)/2 - (self.a - 1)*(self.h + log(xi) + 1) + (self.a - xi)*log(abs(self.a - xi)))*abs(self.a - xi))/((self.a - 1)*abs(self.a - xi)))
                else:
                    self._dydx.append(self.cs*(copysign((self.a - xi)**2/2, self.a - xi) + ((xi-self.a)/2 - (self.a - 1)*(self.h + log(xi) + 1) + (self.a - xi)*log(abs(self.a - xi)) + (xi - 1)*log(1 - xi))*abs(self.a - xi))/((self.a - 1)*abs(self.a - xi)))
        return self._dydx
    @property
    def thc(self):
        if self._thc is None:
            self._thc = [atan(dydxi) for dydxi in self.dydx]
        return self._thc
    def __repr__(self):
        return f'<NACA {self.code:s}>'
