from typing import TYPE_CHECKING

from numpy import sqrt
from pygeom.geom3d import Vector

from .panel import Panel
from .panelprofile import PanelProfile

if TYPE_CHECKING:
    from .panelsheet import PanelSheet


class PanelStrip():
    prfa: PanelProfile = None
    prfb: PanelProfile = None
    sht: 'PanelSheet' = None
    pnls: list[Panel] = None
    ind: int = None
    _point: Vector = None
    _bpos: float = None
    _chord: float = None
    _twist: float = None
    _tilt: float = None
    _width: float = None
    _area: float = None

    def __init__(self, prfa: PanelProfile, prfb: PanelProfile, sht: 'PanelSheet') -> None:
        self.prfa = prfa
        self.prfb = prfb
        self.sht = sht

    @property
    def noload(self) -> bool:
        return self.sht.noload

    @property
    def nohsv(self) -> bool:
        return self.sht.nohsv

    def mesh_panels(self, pid: int) -> int:
        num = len(self.prfa.grds)-1
        self.pnls = []
        for i in range(num):
            grd1 = self.prfa.grds[i]
            grd2 = self.prfa.grds[i+1]
            grd3 = self.prfb.grds[i+1]
            grd4 = self.prfb.grds[i]
            grds = [grd1, grd2, grd3, grd4]
            pnl = Panel(pid, grds)
            self.pnls.append(pnl)
            pid += 1
        return pid

    @property
    def point(self) -> Vector:
        if self._point is None:
            self._point = (self.prfa.point + self.prfb.point)/2
        return self._point

    @property
    def xpos(self) -> float:
        return self.point.x

    @property
    def ypos(self) -> float:
        return self.point.y

    @property
    def zpos(self) -> float:
        return self.point.z

    @property
    def bpos(self) -> float:
        if self._bpos is None:
            self._bpos = (self.prfa.bpos + self.prfb.bpos)/2
        return self._bpos

    @property
    def chord(self) -> float:
        if self._chord is None:
            self._chord = (self.prfa.chord + self.prfb.chord)/2
        return self._chord

    @property
    def twist(self) -> float:
        if self._twist is None:
            self._twist = (self.prfa.twist + self.prfb.twist)/2
        return self._twist

    @property
    def tilt(self) -> float:
        if self._tilt is None:
            self._tilt = (self.prfa.tilt + self.prfb.tilt)/2
        return self._tilt

    @property
    def width(self) -> float:
        if self._width is None:
            dy = self.prfb.point.y - self.prfa.point.y
            dz = self.prfb.point.z - self.prfa.point.z
            self._width = sqrt(dy**2 + dz**2)
        return self._width

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.chord*self.width
        return self._area

    @property
    def pind(self) -> list[int]:
        return [pnl.ind for pnl in self.pnls]

    def __str__(self) -> str:
        return f'PanelStrip(prfa={self.prfa}, prfb={self.prfb}, sht={self.sht})'

    def __repr__(self):
        return self.__str__()
