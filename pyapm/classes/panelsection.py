from math import cos, radians
from typing import List, Dict
from numpy import absolute
from pygeom.geom3d import Vector
from pygeom.array3d import zero_arrayvector
from .grid import Grid
from .panel import Panel
from .panelprofile import PanelProfile
from .panelcontrol import PanelControl, panelcontrol_from_dict
from ..tools.airfoil import airfoil_from_dat
from ..tools.naca4 import NACA4

tol = 1e-12

class PanelSection(PanelProfile):
    airfoil: object = None
    bnum: int = None
    bspc: str = None
    mirror: bool = None
    cnum: int = None
    xoc: float = None
    zoc: float = None
    shta: object = None
    shtb: object = None
    grds: List[Grid] = None
    pnls: List[Panel] = None
    ruled: bool = None
    noload: bool = None
    nomesh: bool = None
    ctrls: Dict[str, PanelControl] = None
    _thkcor: float = None
    _scttyp: str = None

    def __init__(self, point: Vector, chord: float, twist: float, airfoil: object):
        super().__init__(point, chord, twist)
        self.point = point
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
        self.mirror = False
        self.bnum = 1
        self.bspc = 'equal'
        self.noload = False
        self.nomesh = False
        self.nohsv = False
        self.xoc = 0.0
        self.zoc = 0.0
        self.ctrls = {}

    def mirror_section_in_y(self, ymir: float=0.0):
        point = Vector(self.point.x, ymir-self.point.y, self.point.z)
        chord = self.chord
        twist = self.twist
        airfoil = self.airfoil
        sect = PanelSection(point, chord, twist, airfoil)
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
        profile = zero_arrayvector((1, num), dtype=float)
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

    def __repr__(self):
        return f'<pyapm.PanelSection at {self.point:}>'

def panelsection_from_json(sectdata: dict) -> PanelSection:
    xpos = sectdata['xpos']
    ypos = sectdata['ypos']
    zpos = sectdata['zpos']
    point = Vector(xpos, ypos, zpos)
    chord = sectdata['chord']
    airfoilstr = sectdata['airfoil']
    if airfoilstr is not None:
        if airfoilstr[-4:] == '.dat':
            airfoil = airfoil_from_dat(airfoilstr)
        elif airfoilstr[0:4].upper() == 'NACA':
            code = airfoilstr[4:].strip()
            if len(code) == 4:
                airfoil = NACA4(code)
        else:
            raise ValueError(f'Airfoil identified by {airfoilstr:s} does not exist.')
    else:
        airfoil = None
    twist = 0.0
    if 'twist' in sectdata:
        twist = sectdata['twist']
    sect = PanelSection(point, chord, twist, airfoil)
    if 'bnum' in sectdata:
        sect.bnum = sectdata['bnum']
    if 'bspc' in sectdata:
        sect.bspc = sectdata['bspc']
    if 'tilt' in sectdata:
        sect.set_tilt(sectdata['tilt'])
    if 'xoc' in sectdata:
        sect.xoc = sectdata['xoc']
    if 'zoc' in sectdata:
        sect.zoc = sectdata['zoc']
    if 'noload' in sectdata:
        sect.noload = sectdata['noload']
    if 'nohsv' in sectdata:
        sect.nohsv = sectdata['nohsv']
    if 'nomesh' in sectdata:
        sect.nomesh = sectdata['nomesh']
        if sect.nomesh:
            sect.noload = True
            # sect.nohsv = True
    if 'controls' in sectdata:
        for name in sectdata['controls']:
            ctrl = panelcontrol_from_dict(name, sectdata['controls'][name])
            sect.add_control(ctrl)
    return sect
