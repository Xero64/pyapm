from typing import TYPE_CHECKING

from matplotlib.pyplot import figure
from numpy import (arctan, asarray, cos, hstack, multiply, pi, power, sin,
                   sqrt, zeros)
from pygeom.tools.spacing import (equal_spacing, full_cosine_spacing,
                                  linear_bias_left)
from scipy.optimize import least_squares, root

from . import read_dat

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PolyFoil():
    name: str = None
    a: list[float] = None
    b0: float = None
    b: list[float] = None
    cspc: str = None
    teclosed: bool = None
    _cdst: list[float] = None
    _xc: list[float] = None
    _yc: list[float] = None
    _dydx: list[float] = None
    _thc: list[float] = None
    _t: list[float] = None
    _dtdx: list[float] = None
    _tht: list[float] = None
    _xu: list[float] = None
    _yu: list[float] = None
    _thu: list[float] = None
    _xl: list[float] = None
    _yl: list[float] = None
    _thl: list[float] = None
    _x: list[float] = None
    _y: list[float] = None
    _th: list[float] = None

    def __init__(self, name: str, a: list[float], b0: float, b: list[float],
                 cnum: int=80, teclosed=False):
        self.name = name
        self.a = a
        self.b0 = b0
        self.b = b
        self.teclosed = teclosed
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
    def cdst(self):
        if self._cdst is None:
            if self.cspc == 'full-cosine':
                self._cdst = full_cosine_spacing(self.cnum)
            elif self.cspc == 'equal':
                self._cdst = equal_spacing(self.cnum)
            else:
                raise ValueError('Incorrect distribution on NACA4')
        return self._cdst

    @property
    def xc(self):
        if self._xc is None:
            self._xc = asarray(linear_bias_left(self.cdst, 0.2))
        return self._xc

    @property
    def yc(self):
        if self._yc is None:
            self._yc = camber(self.xc, self.a)
        return self._yc

    @property
    def dydx(self):
        if self._dydx is None:
            self._dydx = camber_slope(self.xc, self.a)
        return self._dydx

    @property
    def thc(self):
        if self._thc is None:
            self._thc = arctan(self.dydx)
        return self._thc

    @property
    def t(self):
        if self._t is None:
            if self.teclosed:
                sb = self.b0+sum(self.b[0:-1])
                self.b[-1] = -sb
            self._t = thickness(self.xc, self.b0, self.b)/2
        return self._t

    @property
    def dtdx(self):
        if self._dtdx is None:
            if self.teclosed:
                sb = self.b0+sum(self.b[0:-1])
                self.b[-1] = -sb
            self._dtdx = thickness_slope(self.xc, self.b0, self.b)/2
        return self._dtdx

    @property
    def tht(self):
        if self._tht is None:
            self._tht = arctan(self.dtdx)
            self._tht[0] = pi/2
        return self._tht

    @property
    def xu(self):
        if self._xu is None:
            self._xu = [xi-ti*sin(thi) for xi, ti, thi in zip(self.xc, self.t,
                                                              self.thc)]
        return self._xu

    @property
    def yu(self):
        if self._yu is None:
            self._yu = [yi+ti*cos(thi) for yi, ti, thi in zip(self.yc, self.t,
                                                              self.thc)]
        return self._yu

    @property
    def thu(self):
        if self._thu is None:
            self._thu = [thci+thti for thci, thti in zip(self.thc, self.tht)]
        return self._thu

    @property
    def xl(self):
        if self._xl is None:
            self._xl = [xi+ti*sin(thi) for xi, ti, thi in zip(self.xc, self.t,
                                                              self.thc)]
        return self._xl

    @property
    def yl(self):
        if self._yl is None:
            self._yl = [yi-ti*cos(thi) for yi, ti, thi in zip(self.yc, self.t,
                                                              self.thc)]
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

    def plot(self, ax=None):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
            ax.set_aspect('equal')
        ax.plot(self.x, self.y, label=f'{self.name:s}')
        return ax

    def __repr__(self):
        return f'<{self.name:s}>'

def camber(xc: 'NDArray', a: list[float]):
    yc = zeros(xc.shape)
    for i, ai in enumerate(a):
        yc += ai*power(xc, i+1)
    return yc

def camber_slope(xc: 'NDArray', a: list[float]):
    dycdx = zeros(xc.shape)
    for i, ai in enumerate(a):
        dycdx += (i+1)*ai*power(xc, i)
    return dycdx

def thickness(xc: 'NDArray', b0: float, b: list[float]):
    yt = b0*sqrt(xc)
    for i, bi in enumerate(b):
        yt += bi*power(xc, i+1)
    return yt

def thickness_slope(xc: 'NDArray', b0: float, b: list[float]):
    dytdx = b0*power(2*sqrt(xc), -1)
    dytdx[0] = 0.0
    for i, bi in enumerate(b):
        dytdx += (i+1)*bi*power(xc, i)
    return dytdx

def split_xvals(xvals: 'NDArray', nx: int, na: int):
    xc = xvals[:nx]
    nb = len(xvals)
    nb -= nx
    if na is None:
        na = int(nb/2)
        a = xvals[nx:nx+na].tolist()
    elif na == 0:
        a = []
    else:
        a = xvals[nx:nx+na].tolist()
    b = xvals[nx+na:].tolist()
    b0 = b[0]
    b = b[1:]
    return xc, a, b0, b

def fit_func(xvals: 'NDArray', tgt: 'NDArray', coeff: 'NDArray', na: int=0):
    nx = len(coeff)
    xc, a, b0, b = split_xvals(xvals, nx, na)
    yc = camber(xc, a)
    dycdx = camber_slope(xc, a)
    yt = thickness(xc, b0, b)
    thc = arctan(dycdx)
    sinthc = sin(thc)
    costhc = cos(thc)
    to2 = multiply(coeff, yt)/2
    f = xc - multiply(to2, sinthc)
    g = yc + multiply(to2, costhc)
    return hstack((f, g))-tgt

def polyfoil_from_xy(name: str, x: list[float], y: list[float],
                     na: int=None, nb: int=None):
    num = len(x)
    area = 0.0
    for i in range(num):
        xa = x[i-1]
        ya = y[i-1]
        xb = x[i]
        yb = y[i]
        area += yb*xa - xb*ya
    if area > 0.0:
        x.reverse()
        y.reverse()
    xle = min(x)
    ile = x.index(xle)
    yle = y[ile]
    xin = [xi - xle for xi in x]
    yin = [yi - yle for yi in y]
    xl = xin[:ile]
    yl = yin[:ile]
    xu = xin[ile+1:]
    yu = yin[ile+1:]
    nl = len(xl)
    nu = len(xu)
    nx = nl+nu
    ydata = asarray(xl+xu+yl+yu)
    if na is None and nb is None:
        ab = [0.0 for i in range(nl+nu)]
    elif na == 0 and nb is None:
        ab = [0.0 for i in range(nl+nu)]
    else:
        ab = [0.0 for i in range(na+nb+1)]
    xdata = asarray(xl+xu+ab)
    cl = [-1.0 for _ in range(nl)]
    cu = [1.0 for _ in range(nu)]
    coeff = asarray(cl+cu)
    if na is None and nb is None:
        sol = root(fit_func, xdata, args=(ydata, coeff, na))
    elif na == 0 and nb is None:
        sol = root(fit_func, xdata, args=(ydata, coeff, na))
    else:
        sol = least_squares(fit_func, xdata, args=(ydata, coeff, na))
    _, a, b0, b = split_xvals(sol.x, nx, na)
    return PolyFoil(name, a, b0, b)

def polyfoil_from_dat(datfilepath: str, na: int=None, nb: int=None):
    name, x, y = read_dat(datfilepath)
    return polyfoil_from_xy(name, x, y, na=na, nb=nb)
