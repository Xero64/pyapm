from typing import TYPE_CHECKING

from numpy import sqrt
from pygeom.geom3d import Vector

from .panel import Panel
from .wakepanel import TrailingPanel, WakePanel
from .panelprofile import PanelProfile

if TYPE_CHECKING:
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface


class PanelStrip():
    profile_a: PanelProfile = None
    profile_b: PanelProfile = None
    dpanels: list[Panel] = None
    wpanels: list[TrailingPanel | WakePanel] = None
    ind: int = None
    _sheet: 'PanelSheet' = None
    _surface: 'PanelSurface' = None
    _point: Vector = None
    _bpos: float = None
    _chord: float = None
    _twist: float = None
    _tilt: float = None
    _width: float = None
    _area: float = None

    def __init__(self, profile_a: PanelProfile, profile_b: PanelProfile) -> None:
        self.profile_a = profile_a
        self.profile_b = profile_b

    @property
    def sheet(self) -> 'PanelSheet':
        if self._sheet is None:
            if self.profile_a.sheet is self.profile_b.sheet:
                self._sheet = self.profile_a.sheet
            else:
                raise ValueError('PanelStrip profiles belong to different sheets.')
        return self._sheet

    @sheet.setter
    def sheet(self, sheet: 'PanelSheet') -> None:
        self._sheet = sheet

    @property
    def surface(self) -> 'PanelSurface':
        if self._surface is None:
            self._surface = self.sheet.surface
        return self._surface

    @surface.setter
    def surface(self, surface: 'PanelSurface') -> None:
        self._surface = surface

    @property
    def no_load(self) -> bool:
        return self.sheet.no_load

    @property
    def nohsv(self) -> bool:
        return self.sheet.nohsv

    def mesh_panels(self, pid: int) -> int:

        tecloseda = self.profile_a.teclosed
        teclosedb = self.profile_b.teclosed

        if tecloseda or teclosedb:
            num = 2*self.cnum + 1
            beg = 0
            end = num
        else:
            num = 2*self.cnum + 1
            beg = 1
            end = num

        gridsa = self.profile_a.grids[:num + 2]
        gridsb = self.profile_b.grids[:num + 2]

        # Mesh Dirichlet Panels
        self.dpanels = []
        for i in range(beg, end):
            grid1 = gridsa[i]
            grid2 = gridsa[i+1]
            grid3 = gridsb[i+1]
            grid4 = gridsb[i]
            grids = [grid1, grid2, grid3, grid4]
            pnl = Panel(pid, grids)
            self.dpanels.append(pnl)
            pid += 1

        # Mesh Wake Panels
        self.wpanels = []

        if tecloseda or teclosedb:
            pass
        else:
            grid1 = gridsa[0]
            grid2 = gridsa[1]
            grid3 = gridsb[1]
            grid4 = gridsb[0]
            grids = [grid1, grid2, grid3, grid4]
            panel = TrailingPanel(pid, grids)
            # panel.adjpanels = (self.dpanels[0], )
            self.wpanels.append(panel)
            pid += 1

            grid1 = gridsa[-2]
            grid2 = gridsa[-1]
            grid3 = gridsb[-1]
            grid4 = gridsb[-2]
            grids = [grid1, grid2, grid3, grid4]
            panel = TrailingPanel(pid, grids)
            # panel.adjpanels = (self.dpanels[-1], )
            self.wpanels.append(panel)
            pid += 1

        gridas = [self.profile_b.tegrid]
        gridbs = [self.profile_a.tegrid]
        wpanel = WakePanel(pid, gridas, gridbs, dirw=Vector(1.0, 0.0, 0.0))
        # wpanel.adjpanels = (self.dpanels[-1], self.dpanels[0])
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
    def cnum(self) -> int:
        return self.surface.cnum

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
