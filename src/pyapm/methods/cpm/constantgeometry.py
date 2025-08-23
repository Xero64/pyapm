from typing import TYPE_CHECKING, Any

from numpy import cumsum, empty, ndarray, zeros
from pygeom.geom3d import Vector

from .constantgrid import ConstantGrid
from .constantpanel import ConstantPanel
from .constantwakepanel import ConstantWakePanel

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConstantGeometry:
    surfaces: dict[str, 'ConstantSurface']

    __slots__ = tuple(__annotations__)

    def __init__(self):
        self.surfaces = {}

    def add_surface(self, name: str) -> 'ConstantSurface':
        surface = ConstantSurface(name)
        self.surfaces[name] = surface
        return surface

    def get_surface(self, name: str) -> 'ConstantSurface':
        return self.surfaces[name]


class ConstantSurface:
    name: str
    points: Vector
    ppoints: Vector
    pnormals: Vector
    grids: ndarray[Any, 'ConstantGrid']
    panels: ndarray[Any, 'ConstantPanel']
    npanels: list['ConstantPanel']
    wpanels: list['ConstantWakePanel']
    _npoints: Vector
    _ngridindex: 'NDArray'
    _ngridarea: 'NDArray'
    _area: float
    _tegrids: ndarray[Any, 'ConstantGrid']
    _y: 'NDArray'
    _z: 'NDArray'
    _Db: 'NDArray'
    _b: 'NDArray'
    _w: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str) -> None:
        self.name = name
        self.points = None
        self.ppoints = None
        self.pnormals = None
        self.grids = None
        self.panels = None
        self.npanels = None
        self.wpanels = None
        self.reset()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def mesh_grids(self, gid: int) -> int:
        if self.points is None:
            raise ValueError('Points not set.')
        self.grids = empty(self.points.shape, dtype=ConstantGrid)
        for i in range(self.points.shape[0]):
            for j in range(self.points.shape[1]):
                point = self.points[i, j]
                grid = ConstantGrid(gid, *point.to_xyz())
                self.grids[i, j] = grid
                gid += 1
        return gid

    def mesh_panels(self, pid: int, dirl: Vector = Vector(1.0, 0.0, 0.0)) -> int:
        if self.ppoints is None or self.pnormals is None:
            raise ValueError('Panel points or normals not set.')
        self.panels = empty(self.ppoints.shape, dtype=ConstantPanel)
        self.npanels = []
        self.wpanels = []
        for i in range(self.ppoints.shape[0]):
            for j in range(self.ppoints.shape[1]):
                grida = self.grids[i + 1, j]
                gridb = self.grids[i, j]
                gridc = self.grids[i, j + 1]
                gridd = self.grids[i + 1, j + 1]
                npanel = ConstantPanel(pid, grida, gridb, gridc, gridd)
                npanel.point = self.ppoints[i, j]
                npanel.normal = self.pnormals[i, j]
                self.panels[i, j] = npanel
                self.npanels.append(npanel)
                pid += 1
            grida = self.grids[i, -1]
            gridb = self.grids[i + 1, -1]
            wpanel = ConstantWakePanel(pid, grida, gridb, dirl)
            self.wpanels.append(wpanel)
            pid += 1
        return pid

    @property
    def npoints(self) -> Vector:
        if self._npoints is None:
            self._npoints = 0.75*self.points[:, 0] + 0.25*self.points[:, -1]
        return self._npoints

    @property
    def ngridindex(self) -> 'NDArray':
        if self._ngridindex is None:
            self._ngridindex = zeros(self.grids.shape, dtype=int)
            for i in range(self.grids.shape[0]):
                for j in range(self.grids.shape[1]):
                    grid: 'ConstantGrid' = self.grids[i, j]
                    self._ngridindex[i, j] = grid.ind
        return self._ngridindex

    @property
    def ngridarea(self) -> 'NDArray':
        if self._ngridarea is None:
            self._ngridarea = zeros(self.grids.shape, dtype=float)
            for i in range(self.panels.shape[0]):
                for j in range(self.panels.shape[1]):
                    panel: 'ConstantPanel' = self.panels[i, j]
                    gridarea = panel.area/panel.num
                    self._ngridarea[i, j] += gridarea
                    self._ngridarea[i + 1, j] += gridarea
                    self._ngridarea[i, j + 1] += gridarea
                    self._ngridarea[i + 1, j + 1] += gridarea
        return self._ngridarea

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.ngridarea.sum()
        return self._area

    @property
    def tegrids(self) -> ndarray[Any, 'ConstantGrid']:
        if self._tegrids is None:
            self._tegrids = self.grids[:, -1]
        return self._tegrids

    @property
    def y(self) -> 'NDArray':
        if self._y is None:
            self._y = zeros(self.tegrids.size, dtype=float)
            for i in range(self.tegrids.size):
                self._y[i] = getattr(self.tegrids[i], 'y')
        return self._y

    @property
    def z(self) -> 'NDArray':
        if self._z is None:
            self._z = zeros(self.tegrids.size, dtype=float)
            for i in range(self.tegrids.size):
                self._z[i] = getattr(self.tegrids[i], 'z')
        return self._z

    @property
    def Db(self) -> 'NDArray':
        if self._Db is None:
            Dy = self.y[1:] - self.y[:-1]
            Dz = self.z[1:] - self.z[:-1]
            self._Db = (Dy**2 + Dz**2)**0.5
        return self._Db

    @property
    def b(self) -> 'NDArray':
        if self._b is None:
            b = zeros(self.tegrids.size, dtype=float)
            b[1:] = cumsum(self.Db)
            self._b = b - b[-1]/2
        return self._b

    @property
    def w(self) -> 'NDArray':
        if self._w is None:
            self._w = zeros(self.tegrids.size, dtype=float)
            self._w[:-1] = self.Db/2
            self._w[1:] += self.Db/2
        return self._w
