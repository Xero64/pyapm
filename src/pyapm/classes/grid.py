from math import dist
from typing import TYPE_CHECKING

from pygeom.geom3d import Vector
from numpy import asarray, reciprocal, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .panel import Panel


class Grid(Vector):
    gid: int = None
    ind: int = None
    panels: set['Panel'] = None

    def __init__(self, gid: int, x: float, y: float, z: float) -> None:
        self.gid = gid
        super().__init__(x, y, z)
        self.panels = set()

    def __repr__(self) -> str:
        return f'Grid({self.gid}, {self.x}, {self.y}, {self.z})'

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, frm: str) -> str:
        return f'Grid({self.gid:d}, {self.x:{frm}}, {self.y:{frm}}, {self.z:{frm}})'


def grids_parray(grids: list[Grid]) -> 'NDArray':
    num_grids = len(grids)
    max_pind = None
    for grid in grids:
        for panel in grid.panels:
            if max_pind is None or panel.ind > max_pind:
                max_pind = panel.ind
    parray = zeros((num_grids, max_pind + 1), dtype=float)
    for grid in grids:
        distances = []
        indices = []
        for panel in grid.panels:
            distance = (grid - panel.pnto).return_magnitude()
            distances.append(distance)
            indices.append(panel.ind)
        distances = asarray(distances)
        rec_distances = zeros(distances.shape)
        reciprocal(distances, where=distances!=0.0, out=rec_distances)
        check = distances == 0.0
        if check.any():
            rec_distances[check] = 1.0
            rec_distances[~check] = 0.0
        rec_distances_sum = rec_distances.sum()
        weights = rec_distances / rec_distances_sum
        parray[grid.ind, indices] = weights
    return parray
