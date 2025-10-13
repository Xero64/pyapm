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
    sections: list[PanelSection] = None
    functions: dict[str, 'SurfaceFunction'] = None
    close: bool = None
    cnum: int = None
    cspc: str = None
    twist: float = None
    _sheets: list[PanelSheet] = None
    _strips: list[PanelStrip] = None
    _profiles: list[PanelProfile] = None
    _area: float = None
    grids: list[Grid] = None
    dpanels: list[Panel] = None
    wpanels: list[Panel] = None

    def __init__(self, name: str, point: Vector, twist: float, mirror: bool,
                 sections: list[PanelSection], functions: list['SurfaceFunction'], close: bool):
        self.name = name
        self.point = point
        self.twist = twist
        self.sections = sections
        self.mirror = mirror
        self.functions = functions
        self.close = close
        self.update()

    def update(self):
        bval = 0.0
        for i in range(len(self.sections)):
            self.sections[i].bval = bval
            self.sections[i].bpos = bval
            if i < len(self.sections) - 1:
                delx = self.sections[i+1].point.x - self.sections[i].point.x
                dely = self.sections[i+1].point.y - self.sections[i].point.y
                delz = self.sections[i+1].point.z - self.sections[i].point.z
                bval += sqrt(delx**2 + dely**2 + delz**2)
        if self.mirror:
            ymir = self.sections[0].point.y
            sections = [sct.mirror_section_in_y(ymir=ymir) for sct in self.sections]
            sections.reverse()
            self.sections = sections[:-1] + self.sections

    def set_chord_spacing(self, cnum: int):
        self.cnum = cnum
        for section in self.sections:
            section.set_cnum(self.cnum)

    @property
    def sheets(self):
        if self._sheets is None:
            self._sheets = []
            for i in range(len(self.sections) - 1):
                self._sheets.append(PanelSheet(self.sections[i], self.sections[i+1]))
                self._sheets[i].functions = self.functions
        return self._sheets

    @property
    def strips(self):
        if self._strips is None:
            self._strips = []
            for sheet in self.sheets:
                self._strips += sheet.strips
        return self._strips

    @property
    def profiles(self):
        if self._profiles is None:
            self._profiles = []
            for sheet in self.sheets:
                sct1 = sheet.sct1
                sct2 = sheet.sct2
                self._profiles.append(sct1)
                self._profiles += sheet.profiles
            self._profiles.append(sct2)
        return self._profiles

    @property
    def strips_b(self) -> list[float]:
        return [strip.bpos for strip in self.strips]

    @property
    def strips_y(self) -> list[float]:
        return [strip.ypos for strip in self.strips]

    @property
    def strips_z(self) -> list[float]:
        return [strip.zpos for strip in self.strips]

    @property
    def prfb(self) -> list[float]:
        return [prf.bpos for prf in self.profiles]

    @property
    def prfy(self) -> list[float]:
        return [prf.point.y for prf in self.profiles]

    @property
    def prfz(self) -> list[float]:
        return [prf.point.z for prf in self.profiles]

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = 0.0
            for sheet in self.sheets:
                self._area += sheet.area
        return self._area

    def mesh_grids(self, gid: int):
        self.grids = []
        for sheet in self.sheets:
            section_1 = sheet.section_1
            section_2 = sheet.section_2
            gid = section_1.mesh_grids(gid)
            self.grids += section_1.grids
            gid = sheet.mesh_grids(gid)
            self.grids += sheet.grids
        gid = section_2.mesh_grids(gid)
        self.grids += section_2.grids
        return gid

    def mesh_panels(self, pid: int):
        self.dpanels = []
        self.wpanels = []
        for sheet in self.sheets:
            pid = sheet.mesh_panels(pid)
            for panel in sheet.dpanels:
                panel.surface = self
                self.dpanels.append(panel)
            sheet.set_control_panels()
            for panel in sheet.wpanels:
                panel.surface = self
                self.wpanels.append(panel)
        if self.close:
            for sct in self.sections:
                pid = sct.mesh_panels(pid)
                for panel in sct.dpanels:
                    panel.surface = self
                    self.dpanels.append(panel)
        return pid

    @property
    def pinds(self):
        pinds = []
        for panel in self.panels:
            pinds.append(panel.ind)
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
        sections: list[PanelSection] = []
        for sectdata in surfdata['sections']:
            section = PanelSection.from_dict(sectdata, defaults=defaults)
            sections.append(section)
            if section.airfoil is not None:
                section.airfoil.update(cnum)
        # Linear Interpolate Missing Variables
        x, y, z = [], [], []
        c, a, af = [], [], []
        cmb, xoc, zoc = [], [], []
        b = []
        for section in sections:
            x.append(section.point.x)
            y.append(section.point.y)
            z.append(section.point.z)
            c.append(section.chord)
            a.append(section.twist)
            af.append(section.airfoil)
            b.append(section.bpos)
            xoc.append(section.xoc)
            zoc.append(section.zoc)
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
        for i, section in enumerate(sections):
            section.point.x = x[i]
            section.point.y = y[i]
            section.point.z = z[i]
            section.chord = c[i]
            section.twist = a[i]
            section.xoc = xoc[i]
            section.zoc = zoc[i]
            section.airfoil = af[i]
            section.bpos = b[i]
            # section.bval = abs(b[i])
        # Entire Surface Position
        xpos = surfdata.get('xpos', 0.0)
        ypos = surfdata.get('ypos', 0.0)
        zpos = surfdata.get('zpos', 0.0)
        point = Vector(xpos, ypos, zpos)
        twist = surfdata.get('twist', 0.0)
        ruled = surfdata.get('ruled', False)
        for section in sections:
            section.offset_position(xpos, ypos, zpos)
            section.offset_twist(twist)
            section.ruled = ruled
        close = True
        if 'close' in surfdata:
            close = surfdata['close']
        surf = cls(name, point, twist, mirror, sections, funcs, close)
        surf.set_chord_spacing(cnum)
        # Set the span for the surface functions
        bpos = [section.bpos for section in surf.sections]
        bmax = max(bpos)
        bmin = min(bpos)
        brng = bmax - bmin
        for fnc in surf.functions.values():
            fnc.bmax = brng/2
        for section in surf.sections:
            section.bval = abs(section.bpos)
            if 'chord' in surf.functions:
                section.chord = surf.functions['chord'](section.bval)
            if 'twist' in surf.functions:
                section.twist = surf.functions['twist'](section.bval)
            if 'tilt' in surf.functions:
                section.tilt = surf.functions['tilt'](section.bval)
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
