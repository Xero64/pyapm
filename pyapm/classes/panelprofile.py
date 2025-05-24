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
    _crdsys: Coordinate = None
    bval: float = None
    bpos: float = None
    sct1: 'PanelSection' = None
    sct2: 'PanelSection' = None
    grds: list[Grid] = None
    nohsv: bool = None

    def __init__(self, point: Vector, chord: float, twist: float) -> None:
        self.point = point
        self.chord = chord
        self.twist = twist

    def set_tilt(self, tilt: float) -> None:
        self._tilt = tilt

    def set_ruled_twist(self) -> None:
        shpa = self.sct1.get_shape()
        shpb = self.sct2.get_shape()
        shapedir = shpb - shpa
        shp = shpa + self.bval*shapedir
        n = shp.shape[1]
        indle = int((n-1)/2)
        pntle = shp[0, indle]
        pntte = (shp[0, 0]+shp[0, -1])/2
        dirx = (pntte-pntle).to_unit()
        self.twist = -degrees(acos(dirx.x))

    @property
    def tilt(self) -> float:
        if self._tilt is None:
            self._tilt = self.sct2.tilt + self.bval*(self.sct2.tilt - self.sct1.tilt)
        return self._tilt

    @property
    def crdsys(self) -> Coordinate:
        if self._crdsys is None:
            tilt = radians(self.tilt)
            sintilt = sin(tilt)
            costilt = cos(tilt)
            diry = Vector(0.0, costilt, sintilt)
            twist = radians(self.twist)
            sintwist = sin(twist)
            costwist = cos(twist)
            dirx = Vector(costwist, sintwist*sintilt, -sintwist*costilt)
            self._crdsys = Coordinate(self.point, dirx, diry)
        return self._crdsys

    def get_profile(self, offset: bool=True) -> Vector:
        prfa = self.sct1.get_profile(offset=offset)
        prfb = self.sct2.get_profile(offset=offset)
        profiledir = prfb - prfa
        return prfa + self.bval*profiledir

    def get_shape(self) -> Vector:
        profile = self.get_profile()
        scaledprofile = profile*self.chord
        rotatedprofile = self.crdsys.vector_to_global(scaledprofile)
        translatedprofile = rotatedprofile + self.point
        return translatedprofile

    def mesh_grids(self, gid: int) -> int:
        shp = self.get_shape()
        num = shp.shape[1]
        self.grds = []
        te = False
        for i in range(num):
            # te = False
            # if i == 0 or i == num-1:
            #     if not self.nohsv:
            #         te = True
            self.grds.append(Grid(gid, shp[0, i].x, shp[0, i].y, shp[0, i].z, te))
            gid += 1
        if not self.nohsv:
            self.grds[0].te = True
            self.grds[-1].te = True
        return gid

    def __repr__(self) -> str:
        return f'<PanelProfile at {self.point:}>'
