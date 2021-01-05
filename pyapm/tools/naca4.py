from typing import List
from .spacing import full_cosine_spacing, equal_spacing
from .spacing import linear_bias_left
from math import atan, sqrt, pi, sin, cos
from matplotlib.pyplot import figure

class NACA4(object):
    code: str = None
    cnum: int = None
    cspc: str = None
    teclosed: bool = None
    _mt: float = None
    _mc: float = None
    _pc: float = None
    _cdst: List[float] = None
    _xc: List[float] = None
    _yc: List[float] = None
    _dydx: List[float] = None
    _thc: List[float] = None
    _t: List[float] = None
    _dtdx: List[float] = None
    _tht: List[float] = None
    _xu: List[float] = None
    _yu: List[float] = None
    _thu: List[float] = None
    _xl: List[float] = None
    _yl: List[float] = None
    _thl: List[float] = None
    _x: List[float] = None
    _y: List[float] = None
    _th: List[float] = None
    def __init__(self, code: str, cnum: int=80, teclosed=False):
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
    def mt(self):
        if self._mt is None:
            self._mt = float(self.code[2])/10+float(self.code[3])/100
        return self._mt
    @property
    def mc(self):
        if self._mc is None:
            self._mc = float(self.code[0])/100
        return self._mc
    @property
    def pc(self):
        if self._pc is None:
            self._pc = float(self.code[1])/10
        return self._pc
    @property
    def cdst(self):
        if self._cdst is None:
            if self.cspc == 'full-cosine':
                self._cdst = full_cosine_spacing(self.cnum)
            elif self.cspc == 'equal':
                self._cdst = equal_spacing(self.cnum)
            else:
                return ValueError('Incorrect distribution on NACA4')
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
                if xi < self.pc:
                    self._yc.append(self.mc/self.pc**2*(2*self.pc*xi-xi**2))
                else:
                    self._yc.append(self.mc/(1-self.pc)**2*((1-2*self.pc)+2*self.pc*xi-xi**2))
        return self._yc
    @property
    def dydx(self):
        if self._dydx is None:
            self._dydx = []
            for xi in self.xc:
                if xi < self.pc:
                    self._dydx.append(self.mc/self.pc**2*(2*self.pc-2*xi))
                else:
                    self._dydx.append(self.mc/(1-self.pc)**2*(2*self.pc-2*xi))
        return self._dydx
    @property
    def thc(self):
        if self._thc is None:
            self._thc = [atan(dydxi) for dydxi in self.dydx]
        return self._thc
    @property
    def t(self):
        if self._t is None:
            self._t = []
            if self.teclosed:
                for xi in self.xc:
                    self._t.append(self.mt*(1.4845*sqrt(xi)-0.63*xi-1.758*xi**2+1.4215*xi**3-0.518*xi**4))
            else:
                for xi in self.xc:
                    self._t.append(self.mt*(1.4845*sqrt(xi)-0.63*xi-1.758*xi**2+1.4215*xi**3-0.5075*xi**4))
        return self._t
    @property
    def dtdx(self):
        if self._dtdx is None:
            self._dtdx = []
            if self.teclosed:
                for xi in self.xc:
                    if xi == 0.0:
                        self._dtdx.append(0.0)
                    else:
                        self._dtdx.append(self.mt*(0.74225/sqrt(xi)-0.63-3.516*xi+4.2645*xi**2-2.072*xi**3))
            else:
                for xi in self.xc:
                    if xi == 0.0:
                        self._dtdx.append(0.0)
                    else:
                        self._dtdx.append(self.mt*(0.74225/sqrt(xi)-0.63-3.516*xi+4.2645*xi**2-2.03*xi**3))
        return self._dtdx
    @property
    def tht(self):
        if self._tht is None:
            self._tht = [atan(dtdxi) for dtdxi in self.dtdx]
            self._tht[0] = pi/2
        return self._tht
    @property
    def xu(self):
        if self._xu is None:
            self._xu = [xi-ti*sin(thi) for xi, ti, thi in zip(self.xc, self.t, self.thc)]
        return self._xu
    @property
    def yu(self):
        if self._yu is None:
            self._yu = [yi+ti*cos(thi) for yi, ti, thi in zip(self.yc, self.t, self.thc)]
        return self._yu
    @property
    def thu(self):
        if self._thu is None:
            self._thu = [thci+thti for thci, thti in zip(self.thc, self.tht)]
        return self._thu
    @property
    def xl(self):
        if self._xl is None:
            self._xl = [xi+ti*sin(thi) for xi, ti, thi in zip(self.xc, self.t, self.thc)]
        return self._xl
    @property
    def yl(self):
        if self._yl is None:
            self._yl = [yi-ti*cos(thi) for yi, ti, thi in zip(self.yc, self.t, self.thc)]
        return self._yl
    @property
    def thl(self):
        if self._thl is None:
            self._thl = [thci-thti for thci, thti in zip(self.thc, self.tht)]
        return self._thl
    @property
    def x(self):
        if self._x is None:
            self._x = [xi for xi in reversed(self.xl)] + self.xu[1:]
        return self._x
    @property
    def y(self):
        if self._y is None:
            self._y = [yi for yi in reversed(self.yl)] + self.yu[1:]
        return self._y
    @property
    def th(self):
        if self._th is None:
            self._th = [thi-pi for thi in reversed(self.thl)] + self.thu[1:]
        return self._th
    def plot_airfoil(self, ax=None):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
            ax.set_aspect('equal')
        ax.plot(self.x, self.y, label=f'NACA {self.code:s}')
        return ax
    def __repr__(self):
        return f'<NACA {self.code:s}>'
