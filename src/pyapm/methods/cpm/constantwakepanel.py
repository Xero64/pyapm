from pygeom.geom3d import Coordinate, Vector

from .constantgrid import ConstantGrid
from .constantpanel import ConstantPanel


class ConstantWakePanel():
    pid: int = None
    grida: ConstantGrid = None
    gridb: ConstantGrid = None
    dirl: Vector = None
    _vecab: Vector = None
    _coord: Coordinate = None
    _panels: dict[ConstantPanel, bool] = None
    indo: int = None

    def __init__(self, pid: int, grida: ConstantGrid, gridb: ConstantGrid,
                 dirl: Vector, cond: float = 0.0) -> None:
        self.pid = pid
        self.grida = grida
        self.gridb = gridb
        self.dirl = dirl
        self.cond = cond
        self.link()

    def link(self) -> None:
        if isinstance(self.grida, ConstantGrid):
            self.grida.pnls.add(self)
        if isinstance(self.gridb, ConstantGrid):
            self.gridb.pnls.add(self)

    def reset(self) -> None:
        for attr in self.__dict__:
            if not attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def grda(self) -> ConstantGrid:
        return self.grida

    @property
    def grdb(self) -> ConstantGrid:
        return self.gridb

    @property
    def grids(self) -> tuple[ConstantGrid, ConstantGrid]:
        return self.grida, self.gridb

    @property
    def vecab(self) -> Vector:
        if self._vecab is None:
            self._vecab = self.gridb - self.grida
        return self._vecab

    @property
    def coord(self) -> Coordinate:
        if self._coord is None:
            point = (self.grida + self.gridb)/2
            dirx = self.dirl
            diry = self.vecab
            self._coord = Coordinate(point, dirx, diry)
        return self._coord

    @property
    def point(self) -> Vector:
        return self.coord.pnt

    @property
    def normal(self) -> Vector:
        return self.coord.dirz

    @property
    def panels(self) -> dict[ConstantPanel, bool]:
        if self._panels is None:
            panelsa = self.grida.pnls
            panelsb = self.gridb.pnls
            panels: set[ConstantPanel] = panelsa.intersection(panelsb)
            if self in panels:
                panels.remove(self)
            self._panels = {}
            for panel in panels:
                for edge in panel.edges:
                    if edge.grda == self.grida and edge.grdb == self.gridb:
                        self._panels[panel] = True
                    elif edge.grda == self.gridb and edge.grdb == self.grida:
                        self._panels[panel] = False
        return self._panels

    @property
    def grid_force(self) -> Vector:
        force = Vector.zeros(2)
        inda = 0
        indb = 1
        force[inda] += self.vecab/3
        force[inda] += self.vecab/6
        force[indb] += self.vecab/6
        force[indb] += self.vecab/3
        return force

    @property
    def grid_index(self) -> tuple[int, ...]:
        return self.grida.ind, self.gridb.ind

    def __repr__(self) -> str:
        return '<ConstantWakePanel>'
