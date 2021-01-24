from math import sqrt
from typing import List
from .panel import Panel
from .panelprofile import PanelProfile

class PanelStrip(object):
    prfa: PanelProfile = None
    prfb: PanelProfile = None
    sht: object = None
    pnls: List[Panel] = None
    ind: int = None
    _ypos: float = None
    _zpos: float = None
    _bpos: float = None
    _chord: float = None
    _twist: float = None
    _tilt: float = None
    _width: float = None
    _area: float = None
    def __init__(self, prfa: PanelProfile, prfb: PanelProfile, sht: object):
        self.prfa = prfa
        self.prfb = prfb
        self.sht = sht
    @property
    def noload(self):
        return self.sht.noload
    def mesh_panels(self, pid: int):
        num = len(self.prfa.grds)-1
        self.pnls = []
        for i in range(num):
            grd1 = self.prfa.grds[i]
            grd2 = self.prfa.grds[i+1]
            grd3 = self.prfb.grds[i+1]
            grd4 = self.prfb.grds[i]
            grds = [grd1, grd2, grd3, grd4]
            pnl = Panel(pid, grds)
            pnl.noload = self.noload
            self.pnls.append(pnl)
            pid += 1
        return pid
    @property
    def ypos(self):
        if self._ypos is None:
            self._ypos = (self.prfa.point.y + self.prfb.point.y)/2
        return self._ypos
    @property
    def zpos(self):
        if self._zpos is None:
            self._zpos = (self.prfa.point.z + self.prfb.point.z)/2
        return self._zpos
    @property
    def bpos(self):
        if self._bpos is None:
            self._bpos = (self.prfa.bpos + self.prfb.bpos)/2
        return self._bpos
    @property
    def chord(self):
        if self._chord is None:
            self._chord = (self.prfa.chord + self.prfb.chord)/2
        return self._chord
    @property
    def twist(self):
        if self._twist is None:
            self._twist = (self.prfa.twist + self.prfb.twist)/2
        return self._twist
    @property
    def tilt(self):
        if self._tilt is None:
            self._tilt = (self.prfa.tilt + self.prfb.tilt)/2
        return self._tilt
    @property
    def width(self):
        if self._width is None:
            dy = self.prfb.point.y - self.prfa.point.y
            dz = self.prfb.point.z - self.prfa.point.z
            self._width = sqrt(dy**2 + dz**2)
        return self._width
    @property
    def area(self):
        if self._area is None:
            self._area = self.chord*self.width
        return self._area
    @property
    def pind(self):
        return [pnl.ind for pnl in self.pnls]
    def __repr__(self):
        return '<pyapm.PanelStrip>'
