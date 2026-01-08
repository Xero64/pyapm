from typing import TYPE_CHECKING, Any

from numpy import absolute, cos, radians
from pygeom.geom3d import Vector

from ..tools.airfoil import Airfoil, airfoil_from_dat
from ..tools.naca4 import NACA4
from .grid import Grid
from .panel import Panel
from .panelcontrol import PanelControl
from .panelprofile import PanelProfile

if TYPE_CHECKING:
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface

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
    no_load: bool = None
    no_mesh: bool = None
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
        self.no_load = False
        self.mirror = False
        self.airfoil = NACA4('0012')
        self.no_mesh = False
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
        sect.no_mesh = self.no_mesh
        sect.no_load = self.no_load
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

    def set_noload(self, no_load: bool) -> None:
        self.no_load = no_load

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
                if self.sheet_b.no_mesh:
                    self._scttyp = 'notip'
                else:
                    self._scttyp = 'begtip'
            elif self.sheet_b is None:
                if self.sheet_a.no_mesh:
                    self._scttyp = 'notip'
                else:
                    self._scttyp = 'endtip'
            else:
                if self.sheet_a.no_mesh and self.sheet_b.no_mesh:
                    self._scttyp = 'notip'
                elif self.sheet_a.no_mesh:
                    self._scttyp = 'begtip'
                elif self.sheet_b.no_mesh:
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
        gid = super().mesh_grids(gid)

        if self.scttyp == 'begtip' or self.scttyp == 'endtip':

            botgrids = self.grids[2:self.cnum + 1]
            topgrids = self.grids[2*self.cnum:self.cnum + 1:-1]

            for botgrid, topgrid in zip(botgrids, topgrids):
                midvec = (botgrid + topgrid)/2
                midgrid = Grid(gid, midvec.x, midvec.y, midvec.z)
                self.grids.append(midgrid)
                gid += 1

        return gid

    def mesh_panels(self, pid: int) -> int:
        mesh = False
        if self.scttyp == 'begtip':
            mesh = True
            reverse = True
        elif self.scttyp == 'endtip':
            mesh = True
            reverse = False
        self.dpanels = []
        if mesh:

            botgrids = self.grids[1:self.cnum + 2]
            topgrids = self.grids[2*self.cnum + 1:self.cnum:-1]
            midgrids: list[Grid] = []
            midgrids.extend(self.grids[2*self.cnum + 2:])
            midgrids.append(self.grids[self.cnum + 1])

            for i in range(self.cnum):
                grds: list[Grid] = []
                grds.append(botgrids[i])
                grds.append(botgrids[i+1])
                grds.append(midgrids[i+1])
                grds.append(midgrids[i])
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

            for i in range(self.cnum):
                grds: list[Grid] = []
                grds.append(midgrids[i])
                grds.append(midgrids[i+1])
                grds.append(topgrids[i+1])
                grds.append(topgrids[i])
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
        no_load = sectdata.get('no_load', defaults.get('no_load', False))
        section = PanelSection(point, chord, twist)
        section.bpos = sectdata.get('bpos', None)
        section.xoc = sectdata.get('xoc', defaults.get('xoc', None))
        section.zoc = sectdata.get('zoc', defaults.get('zoc', None))
        section.set_cdo(cdo)
        section.set_noload(no_load)
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
        if 'no_mesh' in sectdata:
            section.no_mesh = sectdata['no_mesh']
            if section.no_mesh:
                section.no_load = True
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
