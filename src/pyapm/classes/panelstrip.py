from typing import TYPE_CHECKING

from numpy import sqrt
from pygeom.geom3d import Vector

from .panel import Panel
from .wakepanel import WakePanel
from .panelprofile import PanelProfile

if TYPE_CHECKING:
    from .panelsheet import PanelSheet


class PanelStrip():
    profile_a: PanelProfile = None
    profile_b: PanelProfile = None
    sheet: 'PanelSheet' = None
    dpanels: list[Panel] = None
    wpanels: list[WakePanel] = None
    ind: int = None
    _point: Vector = None
    _bpos: float = None
    _chord: float = None
    _twist: float = None
    _tilt: float = None
    _width: float = None
    _area: float = None

    def __init__(self, profile_a: PanelProfile, profile_b: PanelProfile,
                 sheet: 'PanelSheet') -> None:
        self.profile_a = profile_a
        self.profile_b = profile_b
        self.sheet = sheet

    @property
    def noload(self) -> bool:
        return self.sheet.noload

    @property
    def nohsv(self) -> bool:
        return self.sheet.nohsv

    def mesh_panels(self, pid: int) -> int:
        if len(self.profile_a.grids) != len(self.profile_b.grids):
            raise ValueError('The len(profile_a.grids) must equal the len(profile_b.grids).')
        num = len(self.profile_a.grids) - 1

        # Mesh Dirichlet Panels
        self.dpanels = []
        for i in range(num):
            grd1 = self.profile_a.grids[i]
            grd2 = self.profile_a.grids[i+1]
            grd3 = self.profile_b.grids[i+1]
            grd4 = self.profile_b.grids[i]
            grds = [grd1, grd2, grd3, grd4]
            pnl = Panel(pid, grds)
            self.dpanels.append(pnl)
            pid += 1

        # Mesh Wake Panels
        self.wpanels = []
        tecloseda = (self.profile_a.tegrid == self.profile_a.grids[0] and self.profile_a.tegrid == self.profile_a.grids[-1])
        teclosedb = (self.profile_b.tegrid == self.profile_b.grids[0] and self.profile_b.tegrid == self.profile_b.grids[-1])

        if tecloseda and teclosedb:
            self.wpanels.append(WakePanel(pid, self.profile_a.grids, self.profile_b.grids))
            pid += 1
        else:
            gridas = [self.profile_b.grids[0], self.profile_b.tegrid]
            gridbs = [self.profile_a.grids[0], self.profile_a.tegrid]
            wpanela = WakePanel(pid, gridas, gridbs)
            wpanela.adjpanels = (self.dpanels[0])
            self.wpanels.append(wpanela)
            pid += 1
            gridas = [self.profile_a.grids[-1], self.profile_a.tegrid]
            gridbs = [self.profile_b.grids[-1], self.profile_b.tegrid]
            wpanelb = WakePanel(pid, gridas, gridbs)
            wpanelb.adjpanels = (self.dpanels[-1])
            self.wpanels.append(wpanelb)
            pid += 1
            gridas = [self.profile_b.tegrid]
            gridbs = [self.profile_a.tegrid]
            wpanel = WakePanel(pid, gridas, gridbs, dirw=Vector(1.0, 0.0, 0.0))
            wpanel.adjpanels = (wpanela, wpanelb)
            self.wpanels.append(wpanel)
            pid += 1
        return pid

    @property
    def point(self) -> Vector:
        if self._point is None:
            self._point = (self.profile_a.point + self.profile_b.point)/2
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
            self._bpos = (self.profile_a.bpos + self.profile_b.bpos)/2
        return self._bpos

    @property
    def chord(self) -> float:
        if self._chord is None:
            self._chord = (self.profile_a.chord + self.profile_b.chord)/2
        return self._chord

    @property
    def twist(self) -> float:
        if self._twist is None:
            self._twist = (self.profile_a.twist + self.profile_b.twist)/2
        return self._twist

    @property
    def tilt(self) -> float:
        if self._tilt is None:
            self._tilt = (self.profile_a.tilt + self.profile_b.tilt)/2
        return self._tilt

    @property
    def width(self) -> float:
        if self._width is None:
            dy = self.profile_b.point.y - self.profile_a.point.y
            dz = self.profile_b.point.z - self.profile_a.point.z
            self._width = sqrt(dy**2 + dz**2)
        return self._width

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.chord*self.width
        return self._area

    @property
    def pind(self) -> list[int]:
        return [panel.ind for panel in self.dpanels]

    def __str__(self) -> str:
        return f'PanelStrip(profile_a={self.profile_a}, profile_b={self.profile_b}, sheet={self.sheet})'

    def __repr__(self):
        return self.__str__()
