from typing import TYPE_CHECKING, Any

from numexpr import evaluate
from numpy import arccos, asarray, pi, sqrt
from pygeom.geom1d import CubicSpline1D, LinearSpline1D
from pygeom.geom3d import Vector
from pygeom.tools.spacing import equal_spacing

from ..tools.airfoil import Airfoil, airfoil_interpolation
from ..tools.naca4 import NACA4
from .grid import Grid
from .panel import Panel
from .panelprofile import PanelProfile
from .panelsection import PanelSection
from .panelsheet import PanelSheet
from .panelstrip import PanelStrip

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AirfoilLike = Airfoil | NACA4


class PanelSurface():
    name: str = None
    point: Vector = None
    twist: float = None
    mirror: bool = None
    scts: list[PanelSection] = None
    fncs: dict[str, 'SurfaceFunction'] = None
    close: bool = None
    cnum: int = None
    cspc: str = None
    twist: float = None
    _shts: list[PanelSheet] = None
    _strps: list[PanelStrip] = None
    _prfs: list[PanelProfile] = None
    _area: float = None
    grds: list[Grid] = None
    pnls: list[Panel] = None

    def __init__(self, name: str, point: Vector, twist: float, mirror: bool,
                 scts: list[PanelSection], fncs: list['SurfaceFunction'], close: bool):
        self.name = name
        self.point = point
        self.twist = twist
        self.scts = scts
        self.mirror = mirror
        self.fncs = fncs
        self.close = close
        self.update()

    def update(self):
        bval = 0.0
        for i in range(len(self.scts)):
            self.scts[i].bval = bval
            self.scts[i].bpos = bval
            if i < len(self.scts)-1:
                delx = self.scts[i+1].point.x - self.scts[i].point.x
                dely = self.scts[i+1].point.y - self.scts[i].point.y
                delz = self.scts[i+1].point.z - self.scts[i].point.z
                bval += sqrt(delx**2 + dely**2 + delz**2)
        # for fnc in self.fncs.values():
        #     fnc.set_spline(bval)
        #     if fnc.var == 'twist':
        #         for sct in self.scts:
        #             sct.twist = fnc.interpolate(sct.bval)
        #     if fnc.var == 'chord':
        #         for sct in self.scts:
        #             sct.chord = fnc.interpolate(sct.bval)
        #     if fnc.var == 'tilt':
        #         for sct in self.scts:
        #             sct.set_tilt(fnc.interpolate(sct.bval))
        if self.mirror:
            ymir = self.scts[0].point.y
            scts = [sct.mirror_section_in_y(ymir=ymir) for sct in self.scts]
            scts.reverse()
            self.scts = scts[:-1] + self.scts

    def set_chord_spacing(self, cnum: int):
        self.cnum = cnum
        for sct in self.scts:
            sct.set_cnum(self.cnum)

    @property
    def shts(self):
        if self._shts is None:
            self._shts = []
            for i in range(len(self.scts)-1):
                self._shts.append(PanelSheet(self.scts[i], self.scts[i+1]))
                self._shts[i].fncs = self.fncs
        return self._shts

    @property
    def strps(self):
        if self._strps is None:
            self._strps = []
            for sht in self.shts:
                self._strps += sht.strps
        return self._strps

    @property
    def prfs(self):
        if self._prfs is None:
            self._prfs = []
            for sht in self.shts:
                sct1 = sht.sct1
                sct2 = sht.sct2
                self._prfs.append(sct1)
                self._prfs += sht.prfs
            self._prfs.append(sct2)
        return self._prfs

    @property
    def strpb(self) -> list[float]:
        return [strp.bpos for strp in self.strps]

    @property
    def strpy(self) -> list[float]:
        return [strp.ypos for strp in self.strps]

    @property
    def strpz(self) -> list[float]:
        return [strp.zpos for strp in self.strps]

    @property
    def prfb(self) -> list[float]:
        return [prf.bpos for prf in self.prfs]

    @property
    def prfy(self) -> list[float]:
        return [prf.point.y for prf in self.prfs]

    @property
    def prfz(self) -> list[float]:
        return [prf.point.z for prf in self.prfs]

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = 0.0
            for sht in self.shts:
                self._area += sht.area
        return self._area

    def mesh_grids(self, gid: int):
        self.grds = []
        for sht in self.shts:
            sct1 = sht.sct1
            sct2 = sht.sct2
            gid = sct1.mesh_grids(gid)
            self.grds += sct1.grds
            gid = sht.mesh_grids(gid)
            self.grds += sht.grds
        gid = sct2.mesh_grids(gid)
        self.grds += sct2.grds
        return gid

    def mesh_panels(self, pid: int):
        self.pnls = []
        for sht in self.shts:
            pid = sht.mesh_panels(pid)
            for pnl in sht.pnls:
                pnl.srfc = self
                self.pnls.append(pnl)
            sht.set_control_panels()
        if self.close:
            for sct in self.scts:
                pid = sct.mesh_panels(pid)
                for pnl in sct.pnls:
                    pnl.srfc = self
                    self.pnls.append(pnl)
        return pid

    @property
    def pinds(self):
        pinds = []
        for pnl in self.pnls:
            pinds.append(pnl.ind)
        return pinds

    @classmethod
    def from_dict(cls, surfdata: dict[str, Any],
                  display: bool=False) -> 'PanelSurface':
        name = surfdata['name']
        mirror = surfdata.get('mirror', False)
        if 'cnum' in surfdata:
            cnum = surfdata['cnum']
        if display: print(f'Loading Surface: {name:s}')
        # Read in Defaults
        defaults: dict[str, Any] = surfdata.get('defaults', {})
        # Read in Functions
        funcdatas: dict[str, Any] = surfdata.get('functions', {})
        funcs = {}
        for variable, funcdata in funcdatas.items():
            funcs[variable] = SurfaceFunction.from_dict(variable, funcdata)
        # Set defaults for functions to zero
        for var in funcs:
            if var not in defaults:
                defaults[var] = 0.0
        # Read Section Variables
        sects: list[PanelSection] = []
        for sectdata in surfdata['sections']:
            sect = PanelSection.from_dict(sectdata, defaults=defaults)
            sects.append(sect)
            if sect.airfoil is not None:
                sect.airfoil.update(cnum)
        # Linear Interpolate Missing Variables
        x, y, z = [], [], []
        c, a, af = [], [], []
        cmb, xoc, zoc = [], [], []
        b = []
        for sect in sects:
            x.append(sect.point.x)
            y.append(sect.point.y)
            z.append(sect.point.z)
            c.append(sect.chord)
            a.append(sect.twist)
            af.append(sect.airfoil)
            b.append(sect.bpos)
            xoc.append(sect.xoc)
            zoc.append(sect.zoc)
        # Check for None values in the first and last sections
        if y[0] is None or z[0] is None:
            raise ValueError('Need at least ypos or zpos specified in the first section.')
        if y[-1] is None or z[-1] is None:
            raise ValueError('Need at least ypos or zpos specified in the last section.')
        # Check for None values in the middle sections
        checky = True
        checkz = True
        for yi, zi, bi in zip(y, z, b):
            if yi is None and bi is None:
                checky = False
            if zi is None and bi is None:
                checkz = False
        # Interpolate None values in y and z
        if checky:
            linear_interpolate_none(y, z)
        elif checkz:
            linear_interpolate_none(z, y)
        else:
            raise ValueError('Need at least ypos or zpos or bpos specified in sections.')
        # Determine b values from known y and z values
        bcur = 0.0
        ycur = y[0]
        zcur = z[0]
        for i in range(len(b)):
            if b[i] is None:
                ydel = y[i] - ycur
                zdel = z[i] - zcur
                bdel = (ydel**2 + zdel**2)**0.5
                b[i] = bcur + bdel
                bcur = b[i]
                ycur = y[i]
                zcur = z[i]
        # Interpolate None values in x, y, z, c, a, cmb, xoc, zoc
        x = linear_interpolate_none(b, x)
        y = linear_interpolate_none(b, y)
        z = linear_interpolate_none(b, z)
        c = linear_interpolate_none(b, c)
        c = fill_none(c, 1.0)
        a = linear_interpolate_none(b, a)
        a = fill_none(a, 0.0)
        cmb = linear_interpolate_airfoil(b, cmb)
        xoc = linear_interpolate_none(b, xoc)
        xoc = fill_none(xoc, 0.25)
        zoc = linear_interpolate_none(b, zoc)
        zoc = fill_none(zoc, 0.0)
        display = False
        if display:
            print(f'{x = }')
            print(f'{y = }')
            print(f'{z = }')
            print(f'{c = }')
            print(f'{a = }')
            print(f'{cmb = }')
            print(f'{xoc = }')
            print(f'{zoc = }')
        for i, sect in enumerate(sects):
            sect.point.x = x[i]
            sect.point.y = y[i]
            sect.point.z = z[i]
            sect.chord = c[i]
            sect.twist = a[i]
            sect.xoc = xoc[i]
            sect.zoc = zoc[i]
            sect.airfoil = af[i]
            sect.bpos = b[i]
            # sect.bval = abs(b[i])
        # Entire Surface Position
        xpos = surfdata.get('xpos', 0.0)
        ypos = surfdata.get('ypos', 0.0)
        zpos = surfdata.get('zpos', 0.0)
        point = Vector(xpos, ypos, zpos)
        twist = surfdata.get('twist', 0.0)
        ruled = surfdata.get('ruled', False)
        for sect in sects:
            sect.offset_position(xpos, ypos, zpos)
            sect.offset_twist(twist)
            sect.ruled = ruled
        close = True
        if 'close' in surfdata:
            close = surfdata['close']
        surf = cls(name, point, twist, mirror, sects, funcs, close)
        surf.set_chord_spacing(cnum)
        # Set the span for the surface functions
        bpos = [sect.bpos for sect in surf.scts]
        bmax = max(bpos)
        bmin = min(bpos)
        brng = bmax - bmin
        for fnc in surf.fncs.values():
            fnc.bmax = brng/2
        for sect in surf.scts:
            sect.bval = abs(sect.bpos)
            if 'chord' in surf.fncs:
                sect.chord = surf.fncs['chord'](sect.bval)
            if 'twist' in surf.fncs:
                sect.twist = surf.fncs['twist'](sect.bval)
            if 'tilt' in surf.fncs:
                sect.tilt = surf.fncs['tilt'](sect.bval)
        return surf

    def __repr__(self):
        return f'<PanelSurface: {self.name:s}>'


def linear_interpolate_airfoil(x: list[float],
                               af: list['AirfoilLike | None']) -> list['AirfoilLike']:
    newaf = []
    for i, afi in enumerate(af):
        if afi is None:
            a = None
            for j in range(i, -1, -1):
                if af[j] is not None:
                    a = j
                    break
            b = None
            for j in range(i, len(af)):
                if af[j] is not None:
                    b = j
                    break
            if a is None:
                xa = x[0]
                afa = NACA4('0012')
            else:
                xa = x[a]
                afa = af[a]
            if b is None:
                xb = x[-1]
                afb = NACA4('0012')
            else:
                xb = x[b]
                afb = af[b]
            if isinstance(afa, (Airfoil, NACA4)) and isinstance(afb, (Airfoil, NACA4)):
                fac = (x[i] - xa)/(xb - xa)
                afi = airfoil_interpolation(afa, afb, fac)
            else:
                raise ValueError('Cannot interpolate airfoil.')
        newaf.append(afi)
    return newaf

def linear_interpolate_none(x: list[float], y: list[float]) -> list[float]:
    for i, (xi, yi) in enumerate(zip(x, y)):
        if yi is None and xi is None:
            continue
        elif yi is None:
            a = None
            for j in range(i, -1, -1):
                if y[j] is not None:
                    a = j
                    break
            b = None
            for j in range(i, len(y)):
                if y[j] is not None:
                    b = j
                    break
            if a is None or b is None:
                y[i] = None
            else:
                xa, xb = x[a], x[b]
                ya, yb = y[a], y[b]
                y[i] = (yb - ya)/(xb - xa)*(x[i] - xa)+ya
    return y

def fill_none(x: list[float], xval: float) -> list[float]:
    for i, xi in enumerate(x):
        if xi is None:
            x[i] = xval
    return x


class SurfaceFunction():
    variable: str | None = None
    functype: str | None = None
    bmax: float | None = None
    spline: LinearSpline1D | CubicSpline1D | None = None
    expression: str | None = None

    def __init__(self, variable: str, functype: str) -> None:
        self.variable = variable
        self.functype = functype

    def set_spline(self, spacing: str, values: list[float],
                   interp: str = 'linear') -> None:
        values: 'NDArray' = asarray(values)
        if spacing == 'equal':
            num = values.size
            nspc = equal_spacing(num - 1)
        else:
            raise ValueError('Spacing not implemented.')
        if interp == 'linear':
            self.spline = LinearSpline1D(nspc, values)
        elif interp == 'cubic':
            self.spline = CubicSpline1D(nspc, values)
        else:
            raise ValueError('Interpolation not implemented.')

    def set_expression(self, expression: str) -> None:
        self.expression = expression

    def __call__(self, value: float) -> float:
        b = value
        s = b/self.bmax
        if self.functype == 'spline':
            return self.spline.evaluate_points_at_t(s)
        elif self.functype == 'expression':
            th = arccos(s)
            local_dict = {'b': b, 's': s, 'th': th}
            global_dict = {'pi': pi}
            return evaluate(self.expression, local_dict=local_dict,
                            global_dict=global_dict)
        else:
            raise ValueError('Function type not implemented.')

    @classmethod
    def from_dict(cls, variable: str,
                  funcdata: dict[str, Any]) -> 'SurfaceFunction':
        functype = funcdata.get('functype')
        srfcfunc = cls(variable, functype)
        if functype == 'spline':
            splinedata: dict[str, Any] = funcdata.get('spline')
            spacing = splinedata.get('spacing', 'equal')
            interp = splinedata.get('interp', 'linear')
            values = splinedata.get('values')
            srfcfunc.set_spline(spacing, values, interp)
        elif functype == 'expression':
            expression = funcdata.get('expression')
            srfcfunc.set_expression(expression)
        else:
            raise ValueError('Function type not implemented.')
        return srfcfunc
