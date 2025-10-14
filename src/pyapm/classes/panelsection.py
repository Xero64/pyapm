from typing import Any, TYPE_CHECKING

from numpy import absolute, cos, radians
from pygeom.geom3d import Vector

from ..tools.airfoil import Airfoil, airfoil_from_dat
from ..tools.naca4 import NACA4
from .grid import Grid
from .panel import Panel
from .panelcontrol import PanelControl
from .panelprofile import PanelProfile

if TYPE_CHECKING:
    from ..classes.panelsheet import PanelSheet

TOL = 1e-12


class PanelSection(PanelProfile):
    airfoil: Airfoil = None
    bnum: int = None
    bspc: str = None
    mirror: bool = None
    cnum: int = None
    xoc: float = None
    zoc: float = None
    sheet_a: 'PanelSheet' = None
    sheet_b: 'PanelSheet' = None
    grids: list[Grid] = None
    dpanels: list[Panel] = None
    ruled: bool = None
    noload: bool = None
    nomesh: bool = None
    controls: dict[str, PanelControl] = None
    _thkcor: float = None
    _scttyp: str = None

    def __init__(self, point: Vector, chord: float, twist: float) -> None:
        super().__init__(point, chord, twist)
        self.point = point
        self.chord = chord
        self.twist = twist
        self.update()

    def update(self) -> None:
        self.noload = False
        self.mirror = False
        self.airfoil = NACA4('0012')
        self.nomesh = False
        self.nohsv = False
        self.controls = {}
        self.cdo = 0.0

    def mirror_section_in_y(self, ymir: float = 0.0) -> 'PanelSection':
        point = Vector(self.point.x, ymir-self.point.y, self.point.z)
        chord = self.chord
        twist = self.twist
        airfoil = self.airfoil
        sect = PanelSection(point, chord, twist)
        sect.airfoil = airfoil
        sect.mirror = True
        sect.bnum = self.bnum
        sect.bspc = self.bspc
        sect.nomesh = self.nomesh
        sect.noload = self.noload
        sect.nohsv = self.nohsv
        sect.xoc = self.xoc
        sect.zoc = self.zoc
        sect.bval = self.bval
        sect.bpos = -self.bpos
        sect.controls = self.controls
        if self.tilt is not None:
            sect.set_tilt(-self._tilt)
        return sect

    def set_cnum(self, cnum: int) -> None:
        self.cnum = cnum
        self.airfoil.update(self.cnum)

    def offset_position(self, xpos: float, ypos: float, zpos: float) -> None:
        self.point.x = self.point.x + xpos
        self.point.y = self.point.y + ypos
        self.point.z = self.point.z + zpos

    def offset_twist(self, twist: float):
        self.twist = self.twist + twist

    def set_span_equal_spacing(self, bnum: int) -> None:
        self.bnum = bnum
        self.bspc = 'equal'

    def set_span_cosine_spacing(self, bnum: int) -> None:
        self.bnum = bnum
        self.bspc = 'full-cosine'

    def set_span_semi_cosine_spacing(self, bnum: int) -> None:
        self.bnum = bnum
        self.bspc = 'semi-cosine'

    def set_airfoil(self, airfoil: str | None) -> None:
        if airfoil is None:
            self.airfoil = NACA4('0012')
        elif airfoil[-4:].lower() == '.dat':
            self.airfoil = airfoil_from_dat(airfoil)
        elif airfoil[0:4].lower() == 'naca':
            code = airfoil[4:].strip()
            if len(code) == 4:
                self.airfoil = NACA4(code)

    def set_noload(self, noload: bool) -> None:
        self.noload = noload

    def set_cdo(self, cdo: float) -> None:
        self.cdo = cdo

    def add_control(self, control: PanelControl) -> None:
        self.controls[control.name] = control

    @property
    def tilt(self) -> float:
        if self._tilt is None:
            if self.sheet_a is None and self.sheet_b is None:
                pass
            elif self.sheet_b is None:
                self._tilt = self.sheet_a.tilt
            elif self.sheet_a is None:
                self._tilt = self.sheet_b.tilt
            else:
                self._tilt = (self.sheet_a.tilt + self.sheet_b.tilt)/2
        return self._tilt

    @property
    def thkcor(self) -> float:
        if self._thkcor is None:
            self._thkcor = 1.0
            if self.sheet_a is not None and self.sheet_b is not None:
                halfdelta = (self.sheet_b.tilt - self.sheet_a.tilt)/2
                self._thkcor = 1.0 / cos(radians(halfdelta))
        return self._thkcor

    @property
    def scttyp(self) -> str:
        if self._scttyp is None:
            if self.sheet_a is None:
                if self.sheet_b.nomesh:
                    self._scttyp = 'notip'
                else:
                    self._scttyp = 'begtip'
            elif self.sheet_b is None:
                if self.sheet_a.nomesh:
                    self._scttyp = 'notip'
                else:
                    self._scttyp = 'endtip'
            else:
                if self.sheet_a.nomesh and self.sheet_b.nomesh:
                    self._scttyp = 'notip'
                elif self.sheet_a.nomesh:
                    self._scttyp = 'begtip'
                elif self.sheet_b.nomesh:
                    self._scttyp = 'endtip'
                else:
                    self._scttyp = 'notip'
        return self._scttyp

    def get_profile(self, offset: bool=True) -> Vector:
        num = self.cnum*2 + 1
        profile = Vector.zeros(num)
        for i in range(self.cnum + 1):
            n = num - i - 1
            j = n - num
            profile[i] = Vector(self.airfoil.xl[j], 0.0, self.airfoil.yl[j])
            profile[n] = Vector(self.airfoil.xu[j], 0.0, self.airfoil.yu[j])
        profile.z[absolute(profile.z) < TOL] = 0.0
        profile.z = profile.z*self.thkcor
        if offset:
            offvec = Vector(self.xoc, 0.0, self.zoc)
            profile = profile - offvec
        return profile

    def mesh_grids(self, gid: int) -> int:
        shape = self.get_shape()
        num = shape.size
        tip_te_closed = False
        if self.scttyp == 'begtip' or self.scttyp == 'endtip':
            vec = shape[-1] - shape[0]
            if vec.return_magnitude() < 1e-12:
                tip_te_closed = True
                num -= 1
        self.grids = []
        for i in range(num):
            self.grids.append(Grid(gid, shape[i].x, shape[i].y, shape[i].z))
            gid += 1
        if tip_te_closed:
            self.grids.append(self.grids[0])

        # Mesh Trailing Edge Grid
        tevec = (shape[0] + shape[-1])/2
        self.tegrid = Grid(gid, tevec.x, tevec.y, tevec.z)
        gid += 1

        return gid

    def mesh_panels(self, pid: int) -> int:
        mesh = False
        reverse = False
        if self.scttyp == 'begtip':
            mesh = True
            reverse = True
        elif self.scttyp == 'endtip':
            mesh = True
        self.dpanels = []
        if mesh:
            numgrd = len(self.grids)
            n = numgrd-1
            numpnl = int(n/2)
            for i in range(numpnl):
                grds: list[Grid] = []
                grds.append(self.grids[i])
                grds.append(self.grids[i+1])
                grds.append(self.grids[n-i-1])
                grds.append(self.grids[n-i])
                dist = (grds[0] - grds[-1]).return_magnitude()
                if dist < TOL:
                    grds = grds[:-1]
                if reverse:
                    grds.reverse()
                pnlgrds = []
                for grd in grds:
                    if grd not in pnlgrds:
                        pnlgrds.append(grd)
                pnl = Panel(pid, pnlgrds)
                pnl.section = self
                self.dpanels.append(pnl)
                pid += 1
        return pid

    @classmethod
    def from_dict(cls, sectdata: dict[str, Any],
                  defaults: dict[str, Any]) -> 'PanelSection':
        """Create a LatticeSection object from a dictionary."""
        xpos = sectdata.get('xpos', None)
        ypos = sectdata.get('ypos', None)
        zpos = sectdata.get('zpos', None)
        point = Vector(xpos, ypos, zpos)
        chord = sectdata.get('chord', defaults.get('chord', None))
        twist = sectdata.get('twist', defaults.get('twist', None))
        airfoil = sectdata.get('airfoil', defaults.get('airfoil', None))
        cdo = sectdata.get('cdo', defaults.get('cdo', 0.0))
        noload = sectdata.get('noload', defaults.get('noload', False))
        section = PanelSection(point, chord, twist)
        section.bpos = sectdata.get('bpos', None)
        section.xoc = sectdata.get('xoc', defaults.get('xoc', None))
        section.zoc = sectdata.get('zoc', defaults.get('zoc', None))
        section.set_cdo(cdo)
        section.set_noload(noload)
        section.set_airfoil(airfoil)
        if 'bnum' in sectdata and 'bspc' in sectdata:
            bnum = sectdata['bnum']
            bspc = sectdata['bspc']
            if bspc == 'equal':
                section.set_span_equal_spacing(bnum)
            elif bspc in ('full-cosine', 'cosine'):
                section.set_span_cosine_spacing(bnum)
            elif bspc == 'semi-cosine':
                section.set_span_semi_cosine_spacing(bnum)
        if 'tilt' in sectdata:
            section.set_tilt(sectdata['tilt'])
        if 'nohsv' in sectdata:
            section.nohsv = sectdata['nohsv']
        if 'nomesh' in sectdata:
            section.nomesh = sectdata['nomesh']
            if section.nomesh:
                section.noload = True
                # section.nohsv = True
        if 'controls' in sectdata:
            for name in sectdata['controls']:
                control = PanelControl.from_dict(name, sectdata['controls'][name])
                section.add_control(control)
        return section

    def __str__(self) -> str:
        return f'PanelSection({self.point}, {self.chord}, {self.twist})'

    def __repr__(self) -> str:
        return self.__str__()
