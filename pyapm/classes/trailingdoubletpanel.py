from typing import Tuple
from pygeom.geom3d import Vector
from .grid import Grid
from .trailingdoublet import TrailingDoublet

class TrailingDoubletPanel(TrailingDoublet):
    pid: int = None
    _indd: Tuple[int] = None
    def __init__(self, pid: int, grda: Grid, grdb: Grid, dirl: Vector):
        self.pid = pid
        super().__init__(grda, grdb, dirl)
    @property
    def indd(self) -> Tuple[int]:
        if self._indd is None:
            self._indd = (self.grda.ind, self.grdb.ind)
        return self._indd
