from typing import TYPE_CHECKING

from numpy import arctan2, degrees
from pygeom.geom3d import IHAT, Vector
from pygeom.tools.spacing import (equal_spacing, full_cosine_spacing,
                                  semi_cosine_spacing)

from .grid import Grid
from .panel import Panel
from .panelcontrol import PanelControl
# from .panelfunction import PanelFunction
from .panelprofile import PanelProfile
from .panelsection import PanelSection
from .panelstrip import PanelStrip

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .panelsurface import SurfaceFunction


class PanelSheet():
    sct1: PanelSection = None
    sct2: PanelSection = None
    _ruled: bool = None
    _noload: bool = None
    _nomesh: bool = None
    _nohsv: bool = None
    _bnum: int = None
    _bspc: str = None
    _bdst: list[float] = None
    _prfs: list[PanelProfile] = None
    _strps: list[PanelStrip] = None
    _mirror: bool = None
    _tilt: float = None
    _area: float = None
    grds: list[Grid] = None
    pnls: list[Panel] = None
    _ctrls: dict[str, PanelControl] = None
    fncs: dict[str, 'SurfaceFunction'] = None

    def __init__(self, sct1: PanelSection, sct2: PanelSection) -> None:
        self.sct1 = sct1
        self.sct1.shtb = self
        self.sct2 = sct2
        self.sct2.shta = self
        self.fncs = {}

    @property
    def mirror(self) -> bool:
        if self._mirror is None:
            self._mirror = self.sct1.mirror
        return self._mirror

    @property
    def ruled(self) -> bool:
        if self._ruled is None:
            if self.mirror:
                self._ruled = self.sct2.ruled
            else:
                self._ruled = self.sct1.ruled
        return self._ruled

    @property
    def noload(self) -> bool:
        if self._noload is None:
            if self.mirror:
                self._noload = self.sct2.noload
            else:
                self._noload = self.sct1.noload
        return self._noload

    @property
    def nomesh(self) -> bool:
        if self._nomesh is None:
            if self.mirror:
                self._nomesh = self.sct2.nomesh
            else:
                self._nomesh = self.sct1.nomesh
        return self._nomesh

    @property
    def nohsv(self) -> bool:
        if self._nohsv is None:
            if self.mirror:
                self._nohsv = self.sct2.nohsv
            else:
                self._nohsv = self.sct1.nohsv
        return self._nohsv

    @property
    def ctrls(self):
        if self._ctrls is None:
            self._ctrls = {}
            sct1 = self.sct1
            sct2 = self.sct2
            if self.mirror:
                for control in sct2.ctrls:
                    ctrl = sct2.ctrls[control]
                    newctrl = ctrl.duplicate(mirror=True)
                    self._ctrls[control] = newctrl
            else:
                for control in sct1.ctrls:
                    ctrl = sct1.ctrls[control]
                    newctrl = ctrl.duplicate(mirror=False)
                    self._ctrls[control] = newctrl
            for control in self.ctrls:
                ctrl = self._ctrls[control]
                if ctrl.uhvec.return_magnitude() == 0.0:
                    pntal = Vector((ctrl.xhinge - sct1.xoc)*sct1.chord,
                                   0.0, -sct1.zoc*sct1.chord)
                    pnta = sct1.point + sct1.crdsys.vector_to_global(pntal)
                    pntbl = Vector((ctrl.xhinge-sct2.xoc)*sct2.chord,
                                   0.0, -sct2.zoc*sct2.chord)
                    pntb = sct2.point + sct2.crdsys.vector_to_global(pntbl)
                    hvec = pntb - pnta
                    ctrl.set_hinge_vector(hvec)
        return self._ctrls

    @property
    def bnum(self) -> int:
        if self._bnum is None:
            if self.mirror:
                self._bnum = self.sct2.bnum
            else:
                self._bnum = self.sct1.bnum
        return self._bnum

    @property
    def bspc(self) -> int:
        if self._bspc is None:
            if self.mirror:
                self._bspc = self.sct2.bspc
            else:
                self._bspc = self.sct1.bspc
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
    def prfs(self):
        if self._prfs is None:
            self._prfs = []
            if not self.nomesh:
                pointdir = self.sct2.point - self.sct1.point
                for bd in self.bdst[1:-1]:
                    point = self.sct1.point + bd*pointdir
                    bpos = self.sct1.bpos*(1.0 - bd) + self.sct2.bpos*bd
                    bval = self.sct1.bval*(1.0 - bd) + self.sct2.bval*bd

                    if 'chord' in self.fncs:
                        chord = self.fncs['chord'](bval)
                    else:
                        chord = self.sct1.chord*(1.0 - bd) + self.sct2.chord*bd
                    if 'twist' in self.fncs:
                        twist = self.fncs['twist'](bval)
                    else:
                        twist = self.sct1.twist*(1.0 - bd) + self.sct2.twist*bd
                    if 'tilt' in self.fncs:
                        tilt = self.fncs['tilt'](bval)
                    else:
                        tilt = self.sct1.tilt*(1.0 - bd) + self.sct2.tilt*bd

                    prf = PanelProfile(point, chord, twist)
                    prf.set_tilt(tilt)
                    prf.sct1 = self.sct1
                    prf.sct2 = self.sct2
                    prf.bval = bd
                    prf.bpos = bpos
                    prf.nohsv = self.nohsv
                    if self.ruled:
                        prf.set_ruled_twist()
                    self._prfs.append(prf)
        return self._prfs

    @property
    def strps(self):
        if self._strps is None:
            self._strps = []
            if not self.nomesh:
                if len(self.prfs) == 0:
                    self._strps.append(PanelStrip(self.sct1, self.sct2, self))
                else:
                    self._strps.append(PanelStrip(self.sct1, self.prfs[0], self))
                    for i in range(len(self.prfs)-1):
                        self._strps.append(PanelStrip(self.prfs[i], self.prfs[i+1], self))
                    self._strps.append(PanelStrip(self.prfs[-1], self.sct2, self))
        return self._strps

    @property
    def tilt(self):
        if self._tilt is None:
            dz = self.sct2.point.z - self.sct1.point.z
            dy = self.sct2.point.y - self.sct1.point.y
            self._tilt = degrees(arctan2(dz, dy))
        return self._tilt

    @property
    def area(self):
        if self._area is None:
            bmin = min(self.sct1.bval, self.sct2.bval)
            bmax = max(self.sct1.bval, self.sct2.bval)
            brng = bmax-bmin
            chorda = self.sct1.chord
            chordb = self.sct2.chord
            self._area = (chorda+chordb)/2*brng
        return self._area

    def mesh_grids(self, gid: int):
        self.grds = []
        if not self.nomesh:
            for prf in self.prfs:
                gid = prf.mesh_grids(gid)
                self.grds += prf.grds
        return gid

    def mesh_panels(self, pid: int):
        self.pnls = []
        if not self.nomesh:
            for strp in self.strps:
                pid = strp.mesh_panels(pid)
                for pnl in strp.pnls:
                    pnl.sht = self
                    self.pnls.append(pnl)
        return pid

    def inherit_controls(self):
        self.ctrls = {}
        if self.mirror:
            for control in self.sct2.ctrls:
                ctrl = self.sct2.ctrls[control]
                newctrl = ctrl.duplicate(mirror=True)
                self.ctrls[control] = newctrl
        else:
            for control in self.sct1.ctrls:
                ctrl = self.sct1.ctrls[control]
                newctrl = ctrl.duplicate(mirror=False)
                self.ctrls[control] = newctrl
        for control in self.ctrls:
            ctrl = self.ctrls[control]
            if ctrl.uhvec.return_magnitude() == 0.0:
                pnt1 = self.sct1.pnt
                crd1 = self.sct1.chord
                pnta = pnt1+crd1*IHAT.dot(ctrl.xhinge)
                pnt2 = self.sct2.pnt
                crd2 = self.sct2.chord
                pntb = pnt2+crd2*IHAT.dot(ctrl.xhinge)
                hvec = pntb-pnta
                ctrl.set_hinge_vector(hvec)

    def set_control_panels(self):
        for control in self.ctrls:
            ctrl = self.ctrls[control]
            if self.mirror:
                sct = self.sct2
            else:
                sct = self.sct1
            prf = sct.get_profile(offset=False)
            beg = None
            end = 0
            for i in range(prf.x.shape[1]):
                if prf.x[0, i] < ctrl.xhinge and beg is None:
                    beg = i - 1
                    end = None
                if prf.x[0, i] > ctrl.xhinge and end is None:
                    end = i
            for strp in self.strps:
                for i, pnl in enumerate(strp.pnls):
                    if i < beg or i >= end:
                        ctrl.add_panel(pnl)

    def __repr__(self):
        return '<PanelSheet>'
