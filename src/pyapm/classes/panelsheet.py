from typing import TYPE_CHECKING

from numpy import arctan2, degrees
from pygeom.geom3d import IHAT, Vector
from pygeom.tools.spacing import (equal_spacing, full_cosine_spacing,
                                  semi_cosine_spacing)

from .panelprofile import PanelProfile
from .panelstrip import PanelStrip

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .grid import Grid
    from .panel import Panel
    from .panelcontrol import PanelControl
    from .panelsection import PanelSection
    from .panelsurface import SurfaceFunction
    from .wakepanel import WakePanel


class PanelSheet():
    section_1: 'PanelSection' = None
    section_2: 'PanelSection' = None
    _ruled: bool = None
    _noload: bool = None
    _nomesh: bool = None
    _nohsv: bool = None
    _bnum: int = None
    _bspc: str = None
    _bdst: list[float] = None
    _profiles: list[PanelProfile] = None
    _strips: list[PanelStrip] = None
    _mirror: bool = None
    _tilt: float = None
    _area: float = None
    _controls: dict[str, 'PanelControl'] = None
    grids: list['Grid'] = None
    dpanels: list['Panel'] = None
    wpanels: list['WakePanel'] = None
    functions: dict[str, 'SurfaceFunction'] = None

    def __init__(self, section_1: 'PanelSection', section_2: 'PanelSection') -> None:
        self.section_1 = section_1
        self.section_1.sheet_b = self
        self.section_2 = section_2
        self.section_2.sheet_a = self
        self.functions = {}

    @property
    def mirror(self) -> bool:
        if self._mirror is None:
            self._mirror = self.section_1.mirror
        return self._mirror

    @property
    def ruled(self) -> bool:
        if self._ruled is None:
            if self.mirror:
                self._ruled = self.section_2.ruled
            else:
                self._ruled = self.section_1.ruled
        return self._ruled

    @property
    def noload(self) -> bool:
        if self._noload is None:
            if self.mirror:
                self._noload = self.section_2.noload
            else:
                self._noload = self.section_1.noload
        return self._noload

    @property
    def nomesh(self) -> bool:
        if self._nomesh is None:
            if self.mirror:
                self._nomesh = self.section_2.nomesh
            else:
                self._nomesh = self.section_1.nomesh
        return self._nomesh

    @property
    def nohsv(self) -> bool:
        if self._nohsv is None:
            if self.mirror:
                self._nohsv = self.section_2.nohsv
            else:
                self._nohsv = self.section_1.nohsv
        return self._nohsv

    @property
    def controls(self) -> dict[str, 'PanelControl']:
        if self._controls is None:
            self._controls = {}
            section_1 = self.section_1
            section_2 = self.section_2
            if self.mirror:
                for control in section_2.controls.values():
                    newctrl = control.duplicate(mirror=True)
                    self._controls[control.name] = newctrl
            else:
                for control in section_1.controls.values():
                    newctrl = control.duplicate(mirror=False)
                    self._controls[control.name] = newctrl
            for control in self._controls.values():
                if control.uhvec.return_magnitude() == 0.0:
                    pntal = Vector((control.xhinge - section_1.xoc)*section_1.chord,
                                   0.0, -section_1.zoc*section_1.chord)
                    pnta = section_1.point + section_1.coord.vector_to_global(pntal)
                    pntbl = Vector((control.xhinge-section_2.xoc)*section_2.chord,
                                   0.0, -section_2.zoc*section_2.chord)
                    pntb = section_2.point + section_2.coord.vector_to_global(pntbl)
                    hvec = pntb - pnta
                    control.set_hinge_vector(hvec)
        return self._controls

    @property
    def bnum(self) -> int:
        if self._bnum is None:
            if self.mirror:
                self._bnum = self.section_2.bnum
            else:
                self._bnum = self.section_1.bnum
        return self._bnum

    @property
    def bspc(self) -> int:
        if self._bspc is None:
            if self.mirror:
                self._bspc = self.section_2.bspc
            else:
                self._bspc = self.section_1.bspc
        return self._bspc

    @property
    def bdst(self) -> 'NDArray':
        if self._bdst is None:
            if self.bspc == 'equal':
                bdst = equal_spacing(self.bnum)
            elif self.bspc == 'semi-cosine':
                bdst = semi_cosine_spacing(self.bnum)
            elif self.bspc == 'full-cosine':
                bdst = full_cosine_spacing(self.bnum)
            elif self.bspc == 'cosine':
                bdst = full_cosine_spacing(self.bnum)
            if self.mirror:
                self._bdst = [1.0 - bd for bd in bdst]
                self._bdst.reverse()
            else:
                self._bdst = bdst
        return self._bdst

    @property
    def profiles(self) -> list[PanelProfile]:
        if self._profiles is None:
            self._profiles = []
            if not self.nomesh:
                pointdir = self.section_2.point - self.section_1.point
                for bd in self.bdst[1:-1]:
                    point = self.section_1.point + bd*pointdir
                    bpos = self.section_1.bpos*(1.0 - bd) + self.section_2.bpos*bd
                    bval = self.section_1.bval*(1.0 - bd) + self.section_2.bval*bd

                    if 'chord' in self.functions:
                        chord = self.functions['chord'](bval)
                    else:
                        chord = self.section_1.chord*(1.0 - bd) + self.section_2.chord*bd
                    if 'twist' in self.functions:
                        twist = self.functions['twist'](bval)
                    else:
                        twist = self.section_1.twist*(1.0 - bd) + self.section_2.twist*bd
                    if 'tilt' in self.functions:
                        tilt = self.functions['tilt'](bval)
                    else:
                        tilt = self.section_1.tilt*(1.0 - bd) + self.section_2.tilt*bd

                    profile = PanelProfile(point, chord, twist)
                    profile.set_tilt(tilt)
                    profile.section_1 = self.section_1
                    profile.section_2 = self.section_2
                    profile.bval = bd
                    profile.bpos = bpos
                    if self.ruled:
                        profile.set_ruled_twist()
                    self._profiles.append(profile)
        return self._profiles

    @property
    def strips(self):
        if self._strips is None:
            self._strips = []
            if not self.nomesh:
                if len(self.profiles) == 0:
                    self._strips.append(PanelStrip(self.section_1, self.section_2, self))
                else:
                    self._strips.append(PanelStrip(self.section_1, self.profiles[0], self))
                    for i in range(len(self.profiles)-1):
                        self._strips.append(PanelStrip(self.profiles[i], self.profiles[i+1], self))
                    self._strips.append(PanelStrip(self.profiles[-1], self.section_2, self))
        return self._strips

    @property
    def tilt(self):
        if self._tilt is None:
            dz = self.section_2.point.z - self.section_1.point.z
            dy = self.section_2.point.y - self.section_1.point.y
            self._tilt = degrees(arctan2(dz, dy))
        return self._tilt

    @property
    def area(self):
        if self._area is None:
            bmin = min(self.section_1.bval, self.section_2.bval)
            bmax = max(self.section_1.bval, self.section_2.bval)
            brng = bmax - bmin
            chorda = self.section_1.chord
            chordb = self.section_2.chord
            self._area = (chorda + chordb) / 2 * brng
        return self._area

    def mesh_grids(self, gid: int) -> int:
        self.grids = []
        if not self.nomesh:
            for profile in self.profiles:
                gid = profile.mesh_grids(gid)
                self.grids += profile.grids
        return gid

    def mesh_panels(self, pid: int) -> int:
        self.dpanels = []
        self.wpanels = []
        if not self.nomesh:
            for strip in self.strips:
                pid = strip.mesh_panels(pid)
                for panel in strip.dpanels:
                    panel.sheet = self
                    self.dpanels.append(panel)
                for panel in strip.wpanels:
                    panel.sheet = self
                    self.wpanels.append(panel)
        return pid

    def inherit_controls(self) -> None:
        self._controls = {}
        if self.mirror:
            for control in self.section_2.controls.values():
                ctrl = self.section_2.controls[control]
                newctrl = ctrl.duplicate(mirror=True)
                self._controls[control.name] = newctrl
        else:
            for control in self.section_1.controls.values():
                ctrl = self.section_1.controls[control]
                newctrl = ctrl.duplicate(mirror=False)
                self._controls[control.name] = newctrl
        for control in self.controls.values():
            if control.uhvec.return_magnitude() == 0.0:
                pnt1 = self.section_1.point
                crd1 = self.section_1.chord
                pnta = pnt1 + crd1 * IHAT.dot(control.xhinge)
                pnt2 = self.section_2.point
                crd2 = self.section_2.chord
                pntb = pnt2 + crd2 * IHAT.dot(control.xhinge)
                hvec = pntb - pnta
                control.set_hinge_vector(hvec)

    def set_control_panels(self):
        for control in self.controls.values():
            if self.mirror:
                section = self.section_2
            else:
                section = self.section_1
            profile = section.get_profile(offset=False)
            beg = None
            end = 0
            for i in range(profile.x.size):
                if profile.x[i] < control.xhinge and beg is None:
                    beg = i - 1
                    end = None
                if profile.x[i] > control.xhinge and end is None:
                    end = i
            for strip in self.strips:
                for i, panel in enumerate(strip.dpanels):
                    if i < beg or i >= end:
                        control.add_panel(panel)

    def __repr__(self):
        return f'PanelSheet(section_1={self.section_1}, section_2={self.section_2})'

    def __str__(self):
        return self.__repr__()
