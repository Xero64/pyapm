from typing import TYPE_CHECKING

from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from .constanttriangle import Triangle


class ConstantEdge():
    grda: Vector
    grdb: Vector
    tria: 'Triangle'
    _vecab: Vector

    __slots__ = tuple(__annotations__)

    def __init__(self, grda: Vector, grdb: Vector, tria: 'Triangle') -> None:
        self.grda = grda
        self.grdb = grdb
        self.tria = tria
        self.reset()

    def reset(self, exclude: set[str] | None = None) -> None:
        if exclude is None:
            exclude = set()
        for attr in self.__slots__:
            if attr not in exclude and attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def grds(self) -> tuple[Vector, Vector]:
        return self.grda, self.grdb

    @property
    def vecab(self) -> Vector:
        if self._vecab is None:
            self._vecab = self.grdb - self.grda
        return self._vecab
