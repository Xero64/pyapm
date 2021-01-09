from matplotlib.pyplot import figure
from pygeom.geom2d import CubicSpline2D, Point2D
from .spacing import full_cosine_spacing, equal_spacing
from typing import List

class Airfoil(object):
    name: str = None
    xin: List[float] = None
    yin: List[float] = None
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
    _xsp: List[float] = None
    _ysp: List[float] = None
    _cdst: List[float] = None
    _spline: CubicSpline2D = None
    _x: List[float] = None
    _y: List[float] = None
    _xu: List[float] = None
    _yu: List[float] = None
    _xl: List[float] = None
    _yl: List[float] = None
    _xc: List[float] = None
    _yc: List[float] = None
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
            pnts = [Point2D(xi, yi) for xi, yi in zip(self.xsp, self.ysp)]
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
                return ValueError('Incorrect distribution on NACA4')
        return self._cdst
    def interpolate_spline(self):
        s = self.spline.arc_length()
        s0 = s[0]
        s1 = s[self.ile]
        s2 = s[-1]
        s3 = [s0 + cd*(s1-s0) for cd in self.cdst]
        s4 = [s1 + cd*(s2-s1) for cd in self.cdst]
        s5 = s3 + s4[1:]
        x, y = self.spline.interpolate_spline_points(s5)
        area = 0.0
        for i in range(len(s5)):
            xa = x[i-1]
            ya = y[i-1]
            xb = x[i]
            yb = y[i]
            ai = yb*xa - xb*ya
            area += ai
        if area > 0.0:
            x.reverse()
            y.reverse()
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
    def plot_airfoil(self, ax=None):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
            ax.set_aspect('equal')
        ax.plot(self.x, self.y, label=self.name)
        return ax

def airfoil_from_dat(datfilepath: str):
    x = []
    y = []
    with open(datfilepath, 'rt') as file:
        for i, line in enumerate(file):
            line = line.rstrip('\n')
            if i == 0:
                name = line.strip()
            else:
                split = line.split()
                if len(split) == 2:
                    x.append(float(split[0]))
                    y.append(float(split[1]))
    return Airfoil(name, x, y)
