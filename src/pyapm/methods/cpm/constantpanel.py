from typing import TYPE_CHECKING

from numpy import arange, sign, zeros
from pygeom.geom3d import Coordinate, Plane, Vector

from .constantedge import ConstantEdge
from .constantgrid import ConstantGrid
from .constanttriangle import Triangle

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConstantPanel():
    pid: int = None
    grids: list[ConstantGrid | Vector] = None
    dirl: Vector = None
    _cond: float = None
    _gridvec: Vector = None
    _trias: list[Triangle] = None
    _edges: list[ConstantEdge] = None
    _point: Vector = None
    _coord: Coordinate = None
    _normal: Vector = None
    _area: float = None
    indo: int = None

    def __init__(self, pid: int, *grids: ConstantGrid,
                 dirl: Vector = Vector(1.0, 0.0, 0.0)) -> None:
        self.pid = pid
        self.grids = grids
        self.dirl = dirl
        self.link()

    def link(self) -> None:
        for grid in self.grids:
            if isinstance(grid, ConstantGrid):
                grid.pnls.add(self)

    def reset(self) -> None:
        for attr in self.__dict__:
            if not attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def cond(self) -> float:
        if self._cond is None:
            self._cond = -1.0
        return self._cond

    @cond.setter
    def cond(self, cond: float) -> None:
        self._cond = sign(cond)
        for tria in self.trias:
            tria.cond = self._cond

    @property
    def num(self) -> int:
        return len(self.grids)

    @property
    def gridvec(self) -> Vector:
        if self._gridvec is None:
            self._gridvec = Vector.from_iter(self.grids)
        return self._gridvec

    @property
    def point(self) -> Vector:
        if self._point is None:
            self._point = self.gridvec.sum()/self.gridvec.size
        return self._point

    @point.setter
    def point(self, point: Vector) -> None:
        self._point = point

    @property
    def normal(self) -> Vector:
        if self._normal is None:
            plane = Plane.from_n_points_best_fit(self.gridvec, True)
            self._normal = plane.nrm
        return self._normal

    @normal.setter
    def normal(self, normal: Vector) -> None:
        self._normal = normal

    @property
    def coord(self) -> Coordinate:
        if self._coord is None:
            plane = Plane.from_n_points_best_fit(self.gridvec, True)
            diry = self.normal.cross(self.dirl)
            dirx = diry.cross(plane.nrm)
            self._crd = Coordinate(self.point, dirx, diry)
        return self._crd

    @property
    def trias(self) -> list[Triangle]:
        if self._trias is None:
            self._trias = []
            for i in range(-1, self.num - 1):
                grida = self.grids[i]
                gridb = self.grids[i + 1]
                tria = Triangle(grida, gridb, self.point)
                # tria.cond = self.cond
                self._trias.append(tria)
        return self._trias

    @property
    def edges(self) -> list[ConstantEdge]:
        if self._edges is None:
            self._edges = []
            for tria in self.trias:
                self._edges.append(tria.edgab)
        return self._edges

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = 0.0
            for tria in self.trias:
                self._area += tria.area
        return self._area

    @property
    def grid_force(self) -> Vector:
        force = Vector.zeros(self.num)
        for edge in self.edges:
            inda = self.grids.index(edge.grda)
            indb = self.grids.index(edge.grdb)
            force[inda] += edge.vecab/3
            force[inda] += edge.vecab/6
            force[indb] += edge.vecab/6
            force[indb] += edge.vecab/3
        return force

    @property
    def grid_index(self) -> tuple[int, ...]:
        return tuple([grid.ind for grid in self.grids])

    @property
    def grid_area(self) -> 'NDArray':
        gridarea = zeros(self.num)
        pointarea = 0.0
        for tria in self.trias:
            for grd in tria.grds:
                try:
                    ind = self.grids.index(grd)
                    gridarea[ind] += tria.area/3
                except ValueError:
                    pointarea += tria.area/3
        gridarea += pointarea/self.num
        return gridarea

    def __repr__(self) -> str:
        return f'<ConstantPanel {self.pid:d}>'

    def __str__(self) -> str:
        return f'ConstantPanel({self.pid:d}, grids={self.grids}, dirl={self.dirl})'


def range_indices(size: int) -> tuple['NDArray', 'NDArray']:
    arng = arange(size)
    brng = arange(1, size + 1)
    brng[-1] = 0
    return arng, brng
