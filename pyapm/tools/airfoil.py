from matplotlib.pyplot import figure
from numpy import asarray
from pygeom.geom2d import Vector2D
from pygeom.geom2d.cubicspline2d import CubicSpline2D
from pygeom.tools.spacing import equal_spacing, full_cosine_spacing

from . import read_dat


class Airfoil():
    name: str = None
    xin: list[float] = None
    yin: list[float] = None
    cnum: int = None
    cspc: str = None
    normalise: bool = None
    shift: bool = None
    _xte: float = None
    _yte: float = None
    _xle: float = None
    _yle: float = None
    _ile: int = None
    _chord: float = None
    _xsp: list[float] = None
    _ysp: list[float] = None
    _cdst: list[float] = None
    _spline: CubicSpline2D = None
    _x: list[float] = None
    _y: list[float] = None
    _xu: list[float] = None
    _yu: list[float] = None
    _xl: list[float] = None
    _yl: list[float] = None
    _xc: list[float] = None
    _yc: list[float] = None

    def __init__(self, name: str, xin: list, yin: list, cnum: int=80,
                 normalise: bool=False, shift: bool=False):
        self.name = name
        self.xin = xin
        self.yin = yin
        self.cnum = cnum
        self.normalise = normalise
        self.shift = shift
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
    def xte(self):
        if self._xte is None:
            self._xte = (self.xin[0]+self.xin[-1])/2
        return self._xte

    @property
    def yte(self):
        if self._yte is None:
            self._yte = (self.yin[0]+self.yin[-1])/2
        return self._yte

    @property
    def xle(self):
        if self._xle is None:
            self._xle = min(self.xin)
        return self._xle

    @property
    def ile(self):
        if self._ile is None:
            self._ile = self.xin.index(self.xle)
        return self._ile

    @property
    def yle(self):
        if self._yle is None:
            self._yle = self.yin[self.ile]
        return self._yle

    @property
    def chord(self):
        if self._chord is None:
            self._chord = self.xte - self.xle
        return self._chord

    @property
    def xsp(self):
        if self._xsp is None:
            if self.normalise:
                if self.shift:
                    self._xsp = [(xi-self.xle)/self.chord for xi in self.xin]
                else:
                    self._xsp = [xi/self.chord for xi in self.xin]
            else:
                if self.shift:
                    self._xsp = [xi-self.xle for xi in self.xin]
                else:
                    self._xsp = [xi for xi in self.xin]
        return self._xsp

    @property
    def ysp(self):
        if self._ysp is None:
            if self.normalise:
                if self.shift:
                    self._ysp = [(yi-self.yle)/self.chord for yi in self.yin]
                else:
                    self._ysp = [yi/self.chord for yi in self.yin]
            else:
                if self.shift:
                    self._ysp = [yi-self.yle for yi in self.yin]
                else:
                    self._ysp = [yi for yi in self.yin]
        return self._ysp

    @property
    def spline(self):
        if self._spline is None:
            xsp = asarray(self.xsp)
            ysp = asarray(self.ysp)
            pnts = Vector2D(xsp, ysp)
            self._spline = CubicSpline2D(pnts)
        return self._spline

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

    def interpolate_spline(self):
        s = self.spline.s
        s0 = s[0]
        s1 = s[self.ile]
        s2 = s[-1]
        s3 = [s0 + cd*(s1-s0) for cd in self.cdst]
        s4 = [s1 + cd*(s2-s1) for cd in self.cdst]
        s5 = s3 + s4[1:]
        x, y = self.spline.evaluate_points_at_t(asarray(s5)).to_xy()
        area = 0.0
        for i in range(len(s5)):
            xa = x[i-1]
            ya = y[i-1]
            xb = x[i]
            yb = y[i]
            ai = yb*xa - xb*ya
            area += ai
        if area > 0.0:
            x = x[::-1]
            y = y[::-1]
        self._x = x
        self._y = y

    @property
    def x(self):
        if self._x is None:
            self.interpolate_spline()
        return self._x

    @property
    def y(self):
        if self._y is None:
            self.interpolate_spline()
        return self._y

    @property
    def xu(self):
        if self._xu is None:
            self._xu = self.x[-self.cnum-1:]
        return self._xu

    @property
    def yu(self):
        if self._yu is None:
            self._yu = self.y[-self.cnum-1:]
        return self._yu

    @property
    def xl(self):
        if self._xl is None:
            self._xl = [xi for xi in reversed(self.x[:self.cnum+1])]
        return self._xl

    @property
    def yl(self):
        if self._yl is None:
            self._yl = [yi for yi in reversed(self.y[:self.cnum+1])]
        return self._yl

    @property
    def xc(self):
        if self._xc is None:
            self._xc = [(xli+xui)/2 for xli, xui in zip(self.xl, self.xu)]
        return self._xc

    @property
    def yc(self):
        if self._yc is None:
            self._yc = [(yli+yui)/2 for yli, yui in zip(self.yl, self.yu)]
        return self._yc

    def plot(self, ax=None):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
            ax.set_aspect('equal')
        ax.plot(self.x, self.y, label=self.name)
        return ax

def airfoil_from_dat(datfilepath: str):
    name, x, y = read_dat(datfilepath)
    return Airfoil(name, x, y)

def airfoil_interpolation(airfoila, airfoilb, fac: float):
    x = [xai*(1-fac) + xbi*fac for xai, xbi in zip(airfoila.x, airfoilb.x)]
    y = [yai*(1-fac) + ybi*fac for yai, ybi in zip(airfoila.y, airfoilb.y)]
    name = 'Interpolated from ' + airfoila.name + ' and ' + airfoilb.name
    return Airfoil(name, x, y)
