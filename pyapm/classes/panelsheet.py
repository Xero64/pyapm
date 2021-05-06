from math import degrees, atan2
from typing import List, Dict
from numpy.matlib import matrix
from pygeom.geom3d import Vector, ihat
from .panelsection import PanelSection
from .panelstrip import PanelStrip
from .panelprofile import PanelProfile
from .panelfunction import PanelFunction
from .panelcontrol import PanelControl
from .grid import Grid
from .panel import Panel
from ..tools import equal_spacing, semi_cosine_spacing, full_cosine_spacing

class PanelSheet(object):
    scta: PanelSection = None
    sctb: PanelSection = None
    fncs: Dict[str, PanelFunction] = None
    _ruled: bool = None
    _noload: bool = None
    _nomesh: bool = None
    _nohsv: bool = None
    _bnum: int = None
    _bspc: str = None
    _bdst: List[float] = None
    _prfs: List[PanelProfile] = None
    _strps: List[PanelStrip] = None
    _mirror: bool = None
    _tilt: float = None
    _area: float = None
    grds: List[Grid] = None
    pnls: List[Panel] = None
    _ctrls: Dict[str, PanelControl] = None
    def __init__(self, scta: PanelSection, sctb: PanelSection):
        self.scta = scta
        self.scta.shtb = self
        self.sctb = sctb
        self.sctb.shta = self
        self.fncs = {}
    @property
    def mirror(self) -> bool:
        if self._mirror is None:
            self._mirror = self.scta.mirror
        return self._mirror
    @property
    def ruled(self) -> bool:
        if self._ruled is None:
            if self.mirror:
                self._ruled = self.sctb.ruled
            else:
                self._ruled = self.scta.ruled
        return self._ruled
    @property
    def noload(self) -> bool:
        if self._noload is None:
            if self.mirror:
                self._noload = self.sctb.noload
            else:
                self._noload = self.scta.noload
        return self._noload
    @property
    def nomesh(self) -> bool:
        if self._nomesh is None:
            if self.mirror:
                self._nomesh = self.sctb.nomesh
            else:
                self._nomesh = self.scta.nomesh
        return self._nomesh
    @property
    def nohsv(self) -> bool:
        if self._nohsv is None:
            if self.mirror:
                self._nohsv = self.sctb.nohsv
            else:
                self._nohsv = self.scta.nohsv
        return self._nohsv
    @property
    def ctrls(self):
        if self._ctrls is None:
            self._ctrls = {}
            scta = self.scta
            sctb = self.sctb
            if self.mirror:
                for control in sctb.ctrls:
                    ctrl = sctb.ctrls[control]
                    newctrl = ctrl.duplicate(mirror=True)
                    self._ctrls[control] = newctrl
            else:
                for control in scta.ctrls:
                    ctrl = scta.ctrls[control]
                    newctrl = ctrl.duplicate(mirror=False)
                    self._ctrls[control] = newctrl
            for control in self.ctrls:
                ctrl = self._ctrls[control]
                if ctrl.uhvec.return_magnitude() == 0.0:
                    pntal = Vector((ctrl.xhinge-scta.xoc)*scta.chord, 0.0, -scta.zoc*scta.chord)
                    pnta = scta.point+scta.crdsys.vector_to_global(pntal)
                    pntbl = Vector((ctrl.xhinge-sctb.xoc)*sctb.chord, 0.0, -sctb.zoc*sctb.chord)
                    pntb = sctb.point+sctb.crdsys.vector_to_global(pntbl)
                    hvec = pntb-pnta
                    ctrl.set_hinge_vector(hvec)
        return self._ctrls
    @property
    def bnum(self) -> int:
        if self._bnum is None:
            if self.mirror:
                self._bnum = self.sctb.bnum
            else:
                self._bnum = self.scta.bnum
        return self._bnum
    @property
    def bspc(self) -> int:
        if self._bspc is None:
            if self.mirror:
                self._bspc = self.sctb.bspc
            else:
                self._bspc = self.scta.bspc
        return self._bspc
    @property
    def bdst(self) -> matrix:
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
                self._bdst = [1.0-bd for bd in bdst]
                self._bdst.reverse()
            else:
                self._bdst = bdst
        return self._bdst
    @property
    def prfs(self):
        if self._prfs is None:
            self._prfs = []
            if not self.nomesh:
                bmin = min(self.scta.bval, self.sctb.bval)
                bmax = max(self.scta.bval, self.sctb.bval)
                brng = bmax-bmin
                pointdir = self.sctb.point-self.scta.point
                for bd in self.bdst[1:-1]:
                    if self.mirror:
                        bint = bmax-bd*brng
                    else:
                        bint = bmin+bd*brng
                    point = self.scta.point + bd*pointdir
                    if 'chord' in self.fncs:
                        chord = self.fncs['chord'].interpolate(bint)
                    else:
                        chord = self.scta.chord*(1.0-bd)+self.sctb.chord*bd
                    if 'twist' in self.fncs:
                        twist = self.fncs['twist'].interpolate(bint)
                    else:
                        twist = self.scta.twist*(1.0-bd)+self.sctb.twist*bd
                    if 'tilt' in self.fncs:
                        tilt = self.fncs['tilt'].interpolate(bint)
                    else:
                        tilt = self.scta.tilt*(1.0-bd)+self.sctb.tilt*bd
                    bpos = self.scta.bpos*(1.0-bd)+self.sctb.bpos*bd
                    prf = PanelProfile(point, chord, twist)
                    prf.set_tilt(tilt)
                    prf.scta = self.scta
                    prf.sctb = self.sctb
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
                    self._strps.append(PanelStrip(self.scta, self.sctb, self))
                else:
                    self._strps.append(PanelStrip(self.scta, self.prfs[0], self))
                    for i in range(len(self.prfs)-1):
                        self._strps.append(PanelStrip(self.prfs[i], self.prfs[i+1], self))
                    self._strps.append(PanelStrip(self.prfs[-1], self.sctb, self))
        return self._strps
    @property
    def tilt(self):
        if self._tilt is None:
            dz = self.sctb.point.z - self.scta.point.z
            dy = self.sctb.point.y - self.scta.point.y
            self._tilt = degrees(atan2(dz, dy))
        return self._tilt
    @property
    def area(self):
        if self._area is None:
            bmin = min(self.scta.bval, self.sctb.bval)
            bmax = max(self.scta.bval, self.sctb.bval)
            brng = bmax-bmin
            chorda = self.scta.chord
            chordb = self.sctb.chord
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
            for control in self.sctb.ctrls:
                ctrl = self.sctb.ctrls[control]
                newctrl = ctrl.duplicate(mirror=True)
                self.ctrls[control] = newctrl
        else:
            for control in self.scta.ctrls:
                ctrl = self.scta.ctrls[control]
                newctrl = ctrl.duplicate(mirror=False)
                self.ctrls[control] = newctrl
        for control in self.ctrls:
            ctrl = self.ctrls[control]
            if ctrl.uhvec.return_magnitude() == 0.0:
                pnt1 = self.scta.pnt
                crd1 = self.scta.chord
                pnta = pnt1+crd1*ihat*ctrl.xhinge
                pnt2 = self.sctb.pnt
                crd2 = self.sctb.chord
                pntb = pnt2+crd2*ihat*ctrl.xhinge
                hvec = pntb-pnta
                ctrl.set_hinge_vector(hvec)
    def set_control_panels(self):
        for control in self.ctrls:
            ctrl = self.ctrls[control]
            if self.mirror:
                sct = self.sctb
            else:
                sct = self.scta
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
