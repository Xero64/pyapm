from typing import TYPE_CHECKING

from numpy import acos, cos, degrees, radians, sin
from pygeom.geom3d import Coordinate, Vector

from .grid import Grid

if TYPE_CHECKING:
    from .panelsection import PanelSection

class PanelProfile():
    point: Vector = None
    chord: float = None
    twist: float = None
    _tilt: float = None
    _coord: Coordinate = None
    bval: float = None
    bpos: float = None
    section_1: 'PanelSection' = None
    section_2: 'PanelSection' = None
    grids: list[Grid] = None
    tegrid: Grid = None

    def __init__(self, point: Vector, chord: float, twist: float) -> None:
        self.point = point
        self.chord = chord
        self.twist = twist

    def set_tilt(self, tilt: float) -> None:
        self._tilt = tilt

    def set_ruled_twist(self) -> None:
        shpa = self.section_1.get_shape()
        shpb = self.section_2.get_shape()
        shapedir = shpb - shpa
        shp = shpa + self.bval*shapedir
        n = shp.size
        indle = int((n-1)/2)
        pntle = shp[indle]
        pntte = (shp[0] + shp[-1])/2
        dirx = (pntte - pntle).to_unit()
        self.twist = -degrees(acos(dirx.x))

    @property
    def tilt(self) -> float:
        if self._tilt is None:
            self._tilt = self.section_2.tilt + self.bval*(self.section_2.tilt - self.section_1.tilt)
        return self._tilt

    @property
    def coord(self) -> Coordinate:
        if self._coord is None:
            tilt = radians(self.tilt)
            sintilt = sin(tilt)
            costilt = cos(tilt)
            diry = Vector(0.0, costilt, sintilt)
            twist = radians(self.twist)
            sintwist = sin(twist)
            costwist = cos(twist)
            dirx = Vector(costwist, sintwist*sintilt, -sintwist*costilt)
            self._coord = Coordinate(self.point, dirx, diry)
        return self._coord

    def get_profile(self, offset: bool=True) -> Vector:
        profile_a = self.section_1.get_profile(offset=offset)
        profile_b = self.section_2.get_profile(offset=offset)
        profiledir = profile_b - profile_a
        return profile_a + self.bval*profiledir

    def get_shape(self) -> Vector:
        profile = self.get_profile()
        scaledprofile = profile*self.chord
        rotatedprofile = self.coord.vector_to_global(scaledprofile)
        translatedprofile = rotatedprofile + self.point
        return translatedprofile

    def mesh_grids(self, gid: int) -> int:
        shape = self.get_shape()
        num = shape.size

        # Mesh Trailing Edge Grid
        tevec = (shape[0] + shape[-1])/2

        # Mesh Profile Grids
        self.grids = []
        self.grids.append(Grid(gid, tevec.x, tevec.y, tevec.z))
        gid += 1
        for i in range(num):
            self.grids.append(Grid(gid, shape[i].x, shape[i].y, shape[i].z))
            gid += 1
        self.grids.append(Grid(gid, tevec.x, tevec.y, tevec.z))
        gid += 1

        self.tegrid = Grid(gid, tevec.x, tevec.y, tevec.z)
        gid += 1

        return gid

    def __repr__(self) -> str:
        return f'PanelProfile(point={self.point}, chord={self.chord}, twist={self.twist})'

    def __str__(self) -> str:
        return self.__repr__()
