from .constantgrid import ConstantGrid
from .constantedge import ConstantEdge


class Triangle():
    grda: ConstantGrid
    grdb: ConstantGrid
    grdc: ConstantGrid
    _edgab: ConstantEdge
    _edgbc: ConstantEdge
    _edgca: ConstantEdge
    _jac: float
    _area: float

    __slots__ = tuple(__annotations__)

    def __init__(self, grda: ConstantGrid, grdb: ConstantGrid, grdc: ConstantGrid) -> None:
        self.grda = grda
        self.grdb = grdb
        self.grdc = grdc
        self.reset()

    def reset(self, exclude: set[str] | None = None) -> None:
        if exclude is None:
            exclude = set()
        for attr in self.__slots__:
            if attr not in exclude and attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def grds(self) -> tuple[ConstantGrid, ConstantGrid, ConstantGrid]:
        return self.grda, self.grdb, self.grdc

    @property
    def edgab(self) -> ConstantEdge:
        if self._edgab is None:
            self._edgab = ConstantEdge(self.grda, self.grdb, self)
        return self._edgab

    @property
    def edgbc(self) -> ConstantEdge:
        if self._edgbc is None:
            self._edgbc = ConstantEdge(self.grdb, self.grdc, self)
        return self._edgbc

    @property
    def edgca(self) -> ConstantEdge:
        if self._edgca is None:
            self._edgca = ConstantEdge(self.grdc, self.grda, self)
        return self._edgca

    @property
    def edgs(self) -> tuple[ConstantEdge, ConstantEdge, ConstantEdge]:
        return self.edgab, self.edgbc, self.edgca

    @property
    def jac(self) -> float:
        if self._jac is None:
            self._jac = self.edgab.vecab.cross(self.edgbc.vecab).return_magnitude()
        return self._jac

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = 0.5 * self.jac
        return self._area
