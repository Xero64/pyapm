from .panel import Panel
from .panelprofile import PanelProfile
from typing import List
from math import sqrt

class PanelStrip(object):
    prfa: PanelProfile = None
    prfb: PanelProfile = None
    sht: object = None
    pnls: List[Panel] = None
    ind: int = None
    _y: float = None
    _z: float = None
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
            gids = [grd1.gid, grd2.gid, grd3.gid, grd4.gid]
            pnl = Panel(pid, gids)
            pnl.noload = self.noload
            self.pnls.append(pnl)
            pid += 1
        return pid
    @property
    def y(self):
        if self._y is None:
            self._y = (self.prfa.point.y + self.prfb.point.y)/2
        return self._y
    @property
    def z(self):
        if self._z is None:
            self._z = (self.prfa.point.z + self.prfb.point.z)/2
        return self._z
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
