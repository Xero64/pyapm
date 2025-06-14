from typing import Any

from numpy import absolute, cos, radians
from pygeom.geom3d import Vector

from ..tools.airfoil import Airfoil, airfoil_from_dat
from ..tools.naca4 import NACA4
from .grid import Grid
from .panel import Panel
from .panelcontrol import PanelControl
from .panelprofile import PanelProfile

tol = 1e-12


class PanelSection(PanelProfile):
    airfoil: Airfoil = None
    bnum: int = None
    bspc: str = None
    mirror: bool = None
    cnum: int = None
    xoc: float = None
    zoc: float = None
    shta: object = None
    shtb: object = None
    grds: list[Grid] = None
    pnls: list[Panel] = None
    ruled: bool = None
    noload: bool = None
    nomesh: bool = None
    ctrls: dict[str, PanelControl] = None
    _thkcor: float = None
    _scttyp: str = None

    def __init__(self, point: Vector, chord: float, twist: float):
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
        self.ctrls = {}
        self.cdo = 0.0

    def mirror_section_in_y(self, ymir: float=0.0):
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
        sect.ctrls = self.ctrls
        if self.tilt is not None:
            sect.set_tilt(-self._tilt)
        return sect

    def set_cnum(self, cnum: int):
        self.cnum = cnum
        self.airfoil.update(self.cnum)

    def offset_position(self, xpos: float, ypos: float, zpos: float):
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

    def add_control(self, ctrl: PanelControl):
        self.ctrls[ctrl.name] = ctrl

    @property
    def tilt(self):
        if self._tilt is None:
            if self.shta is None and self.shtb is None:
                pass
            elif self.shtb is None:
                self._tilt = self.shta.tilt
            elif self.shta is None:
                self._tilt = self.shtb.tilt
            else:
                self._tilt = (self.shta.tilt + self.shtb.tilt)/2
        return self._tilt

    @property
    def thkcor(self):
        if self._thkcor is None:
            self._thkcor = 1.0
            if self.shta is not None and self.shtb is not None:
                halfdelta = (self.shtb.tilt - self.shta.tilt)/2
                self._thkcor = 1.0/cos(radians(halfdelta))
        return self._thkcor

    @property
    def scttyp(self):
        if self._scttyp is None:
            if self.shta is None:
                if self.shtb.nomesh:
                    self._scttyp = 'notip'
                else:
                    self._scttyp = 'begtip'
            elif self.shtb is None:
                if self.shta.nomesh:
                    self._scttyp = 'notip'
                else:
                    self._scttyp = 'endtip'
            else:
                if self.shta.nomesh and self.shtb.nomesh:
                    self._scttyp = 'notip'
                elif self.shta.nomesh:
                    self._scttyp = 'begtip'
                elif self.shtb.nomesh:
                    self._scttyp = 'endtip'
                else:
                    self._scttyp = 'notip'
        return self._scttyp

    def get_profile(self, offset: bool=True):
        num = self.cnum*2+1
        profile = Vector.zeros((1, num))
        for i in range(self.cnum+1):
            n = num-i-1
            j = n-num
            profile[0, i] = Vector(self.airfoil.xl[j], 0.0, self.airfoil.yl[j])
            profile[0, n] = Vector(self.airfoil.xu[j], 0.0, self.airfoil.yu[j])
        profile.z[absolute(profile.z) < tol] = 0.0
        profile.z = profile.z*self.thkcor
        if offset:
            offvec = Vector(self.xoc, 0.0, self.zoc)
            profile = profile-offvec
        return profile

    def mesh_grids(self, gid: int) -> int:
        shp = self.get_shape()
        num = shp.shape[1]
        tip_te_closed = False
        if self.scttyp == 'begtip' or self.scttyp == 'endtip':
            vec = shp[0, -1] - shp[0, 0]
            if vec.return_magnitude() < 1e-12:
                tip_te_closed = True
                num -= 1
        self.grds = []
        te = False
        for i in range(num):
            self.grds.append(Grid(gid, shp[0, i].x, shp[0, i].y, shp[0, i].z, te))
            gid += 1
        if tip_te_closed:
            self.grds.append(self.grds[0])
        if not self.nohsv:
            self.grds[0].te = True
            self.grds[-1].te = True
        return gid

    def mesh_panels(self, pid: int):
        mesh = False
        reverse = False
        if self.scttyp == 'begtip':
            mesh = True
            reverse = True
        elif self.scttyp == 'endtip':
            mesh = True
        self.pnls = []
        if mesh:
            # if self.shta is None:
            #     noload = self.shtb.noload
            # elif self.shtb is None:
            #     noload = self.shta.noload
            # else:
            #     noload = False
            numgrd = len(self.grds)
            n = numgrd-1
            numpnl = int(n/2)
            for i in range(numpnl):
                grds = []
                grds.append(self.grds[i])
                grds.append(self.grds[i+1])
                grds.append(self.grds[n-i-1])
                grds.append(self.grds[n-i])
                dist = (grds[0]-grds[-1]).return_magnitude()
                if dist < tol:
                    grds = grds[:-1]
                if reverse:
                    grds.reverse()
                pnlgrds = []
                for grd in grds:
                    if grd not in pnlgrds:
                        pnlgrds.append(grd)
                pnl = Panel(pid, pnlgrds)
                pnl.sct = self
                self.pnls.append(pnl)
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
        sect = PanelSection(point, chord, twist)
        sect.bpos = sectdata.get('bpos', None)
        sect.xoc = sectdata.get('xoc', defaults.get('xoc', None))
        sect.zoc = sectdata.get('zoc', defaults.get('zoc', None))
        sect.set_cdo(cdo)
        sect.set_noload(noload)
        sect.set_airfoil(airfoil)
        if 'bnum' in sectdata and 'bspc' in sectdata:
            bnum = sectdata['bnum']
            bspc = sectdata['bspc']
            if bspc == 'equal':
                sect.set_span_equal_spacing(bnum)
            elif bspc in ('full-cosine', 'cosine'):
                sect.set_span_cosine_spacing(bnum)
            elif bspc == 'semi-cosine':
                sect.set_span_semi_cosine_spacing(bnum)
        if 'tilt' in sectdata:
            sect.set_tilt(sectdata['tilt'])
        if 'nohsv' in sectdata:
            sect.nohsv = sectdata['nohsv']
        if 'nomesh' in sectdata:
            sect.nomesh = sectdata['nomesh']
            if sect.nomesh:
                sect.noload = True
                # sect.nohsv = True
        if 'controls' in sectdata:
            for name in sectdata['controls']:
                ctrl = PanelControl.from_dict(name, sectdata['controls'][name])
                sect.add_control(ctrl)
        return sect

    def __repr__(self):
        return f'<pyapm.PanelSection at {self.point:}>'
