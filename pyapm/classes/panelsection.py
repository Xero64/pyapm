from math import cos, radians
from typing import List
from numpy.matlib import absolute
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector
from .panel import Panel
from .panelprofile import PanelProfile
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
    pnls: List[Panel] = None
    noload: bool = None
    nomesh: bool = None
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
    def mirror_section_in_y(self, ymir: float=0.0):
        point = Vector(self.point.x, ymir-self.point.y, self.point.z)
        chord = self.chord
        twist = self.twist
        airfoil = self.airfoil
        sect = PanelSection(point, chord, twist, airfoil)
        sect.mirror = True
        sect.bnum = self.bnum
        sect.bspc = self.bspc
        sect.noload = self.noload
        sect.xoc = self.xoc
        sect.zoc = self.zoc
        sect.bval = self.bval
        sect.bpos = -self.bpos
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
        self.twist = self.twist+twist
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
            if self.shta is not None and self.shtb is not None:
                if self.shta.nomesh and self.shtb.nomesh:
                    self._scttyp = 'notip'
            if self.shta is None:
                if not self.shtb.nomesh:
                    self._scttyp = 'begtip'
            elif self.shta.nomesh:
                if not self.shtb.nomesh:
                    self._scttyp = 'begtip'
            if self.shtb is None:
                if not self.shta.nomesh:
                    self._scttyp = 'endtip'
            elif self.shtb.nomesh:
                if not self.shta.nomesh:
                    self._scttyp = 'endtip'
        return self._scttyp
    def get_profile(self):
        num = self.cnum*2+1
        profile = zero_matrix_vector((1, num), dtype=float)
        for i in range(self.cnum+1):
            n = num-i-1
            j = n-num
            profile[0, i] = Vector(self.airfoil.xl[j], 0.0, self.airfoil.yl[j])
            profile[0, n] = Vector(self.airfoil.xu[j], 0.0, self.airfoil.yu[j])
        profile.z[absolute(profile.z) < tol] = 0.0
        profile.z = profile.z*self.thkcor
        offset = Vector(self.xoc, 0.0, self.zoc)
        profile = profile-offset
        return profile
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
            if self.shta is None:
                noload = self.shtb.noload
            elif self.shtb is None:
                noload = self.shta.noload
            else:
                noload = False
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
                pnl.noload = noload
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
    if airfoilstr[-4:] == '.dat':
        airfoil = airfoil_from_dat(airfoilstr)
    elif airfoilstr[0:4].upper() == 'NACA':
        code = airfoilstr[4:].strip()
        if len(code) == 4:
            airfoil = NACA4(code)
    else:
        return ValueError(f'Airfoil identified by {airfoilstr:s} does not exist.')
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
    if 'nomesh' in sectdata:
        sect.nomesh = sectdata['nomesh']
    if 'nohsv' in sectdata:
        sect.nohsv = sectdata['nohsv']
    return sect
