from typing import TYPE_CHECKING

from numpy import cos, sin, zeros
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .constantpanel import ConstantPanel
    from .constantsystem import ConstantSystem
    from .constantgrid import Grid


class ControlObject:
    name: str
    _posgain: float
    _neggain: float
    _position: float
    _vector: Vector
    _point: Vector
    _panels: list['ConstantPanel']
    _panel_index: 'NDArray'
    _grids: list['Grid']
    _grid_index: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self, exclude: set[str] | None = None) -> None:
        if exclude is None:
            exclude = set()
        for attr in self.__slots__:
            if attr not in exclude and attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def posgain(self) -> float:
        if self._posgain is None:
            self._posgain = 1.0
        return self._posgain

    @posgain.setter
    def posgain(self, posgain: float) -> None:
        self._posgain = posgain

    @property
    def neggain(self) -> float:
        if self._neggain is None:
            self._neggain = 1.0
        return self._neggain

    @neggain.setter
    def neggain(self, neggain: float) -> None:
        self._neggain = neggain

    @property
    def position(self) -> float:
        if self._position is None:
            self._position = 0.0
        return self._position

    @position.setter
    def position(self, position: float) -> None:
        self._position = position

    @property
    def vector(self) -> Vector:
        if self._vector is None:
            self._vector = Vector(0.0, 0.0, 0.0)
        return self._vector

    @vector.setter
    def vector(self, vector: Vector) -> None:
        self._vector = vector

    @property
    def point(self) -> Vector:
        if self._point is None:
            self._point = Vector(0.0, 0.0, 0.0)
        return self._point

    @point.setter
    def point(self, point: Vector) -> None:
        self._point = point

    @property
    def panels(self) -> list['ConstantPanel']:
        if self._panels is None:
            self._panels = []
        return self._panels

    def add_panel(self, panel: 'ConstantPanel') -> None:
        self.panels.append(panel)
        self._panel_index = None
        self._grids = None
        self._grid_index = None

    @property
    def panel_index(self) -> 'NDArray':
        if self._panel_index is None:
            self._panel_index = zeros(len(self.panels), dtype=int)
            for i, panel in enumerate(self.panels):
                self._panel_index[i] = panel.indo
        return self._panel_index

    @panel_index.setter
    def panel_index(self, panel_index: 'NDArray') -> None:
        self._panel_index = panel_index

    @property
    def grids(self) -> list['Grid']:
        if self._grids is None:
            self._grids = []
            for panel in self.panels:
                for grid in panel.grids:
                    if grid not in self._grids:
                        self._grids.append(grid)
        return self._grids

    @property
    def grid_index(self) -> 'NDArray':
        if self._grid_index is None:
            self._grid_index = zeros(len(self.grids), dtype=int)
            for i, grid in enumerate(self.grids):
                self._grid_index[i] = grid.ind
        return self._grid_index

    def __str__(self) -> str:
        return f'ControlObject({self.name:s})'

    def __repr__(self) -> str:
        return f'ControlObject({self.name:s})'


class ConstantControl:
    name: str
    system: 'ConstantSystem'
    control_objects: list[ControlObject]
    _hinge_vectors: Vector
    _hinge_points: Vector
    _hinge_normals: Vector
    _index: tuple[int, int]
    _normal_change_approx: Vector

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str, system: 'ConstantSystem') -> None:
        self.name = name
        self.system = system
        self.control_objects = []
        self.reset()

    def reset(self, exclude: set[str] | None = None) -> None:
        if exclude is None:
            exclude = set()
        for attr in self.__slots__:
            if attr not in exclude and attr.startswith('_'):
                setattr(self, attr, None)

    def add_control_object(self, control_object: ControlObject) -> None:
        self.control_objects.append(control_object)

    @property
    def hinge_vectors(self) -> Vector:
        if self._hinge_vectors is None:
            self._hinge_vectors = Vector.zeros((self.system.num_panels, 2))
            for object in self.control_objects:
                for panel in object.panels:
                    self._hinge_vectors[panel.indo, 0] = object.vector*object.posgain
                    self._hinge_vectors[panel.indo, 1] = object.vector*object.neggain
        return self._hinge_vectors

    @hinge_vectors.setter
    def hinge_vectors(self, hinge_vectors: Vector) -> None:
        self._hinge_vectors = hinge_vectors

    @property
    def hinge_points(self) -> Vector:
        if self._hinge_points is None:
            self._hinge_points = Vector.zeros((self.system.num_panels, 1))
            for object in self.control_objects:
                for panel in object.panels:
                    self._hinge_points[panel.indo, 0] = panel.point
        return self._hinge_points

    @hinge_points.setter
    def hinge_points(self, hinge_points: Vector) -> None:
        self._hinge_points = hinge_points

    @property
    def hinge_normals(self) -> Vector:
        if self._hinge_normals is None:
            self._hinge_normals = Vector.zeros(self.system.num_panels)
            for object in self.control_objects:
                for panel in object.panels:
                    self._hinge_normals[panel.indo] = panel.normal
        return self._hinge_normals

    @hinge_normals.setter
    def hinge_normals(self, hinge_normals: Vector) -> None:
        self._hinge_normals = hinge_normals

    @property
    def index(self) -> tuple[int, int]:
        if self._index is None:
            self._index = None
        return self._index

    @index.setter
    def index(self, index: tuple[int, int]) -> None:
        self._index = index

    @property
    def normal_change_approx(self) -> Vector:
        if self._normal_change_approx is None:
            self._normal_change_approx = Vector.zeros(self.hinge_vectors.shape)
            self._normal_change_approx[:, 0] = self.hinge_vectors[:, 0].cross(self.hinge_normals)
            self._normal_change_approx[:, 1] = self.hinge_vectors[:, 1].cross(self.hinge_normals)
        return self._normal_change_approx

    def normal_change_and_derivative(self, angle: float) -> Vector:
        # Rodrigues' Rotation Formula
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        # Angle: Angle of Rotation in Radians
        if angle == 0.0:
            k = self.hinge_vectors.sum(axis=1)/2
        elif angle > 0.0:
            k = self.hinge_vectors[:, 0]
        else:
            k = self.hinge_vectors[:, 1]
        v = self.hinge_normals
        kxv = k.cross(v)
        kdv = k.dot(v)
        cosang = cos(angle)
        sinang = sin(angle)
        vrot = v*cosang + kxv*sinang + k*kdv*(1.0 - cosang)
        vdel = vrot - v
        vder = -v*sinang + kxv*cosang + k*kdv*sinang
        return vdel, vder

    def __str__(self) -> str:
        return f'ConstantControl({self.name:s})'

    def __repr__(self) -> str:
        return f'ConstantControl({self.name:s})'
