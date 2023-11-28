from math import acos, cos, degrees, radians, sin
from typing import TYPE_CHECKING, List

from pygeom.geom3d import Coordinate, Vector
from pygeom.matrix3d import vector_to_global

from .grid import Grid

if TYPE_CHECKING:
    from pygeom.matrix3d import MatrixVector

    from .panelsection import PanelSection

class PanelProfile():
    point: Vector = None
    chord: float = None
    twist: float = None
    _tilt: float = None
    _crdsys: Coordinate = None
    bval: float = None
    bpos: float = None
    scta: 'PanelSection' = None
    sctb: 'PanelSection' = None
    grds: List[Grid] = None
    nohsv: bool = None

    def __init__(self, point: Vector, chord: float, twist: float) -> None:
        self.point = point
        self.chord = chord
        self.twist = twist

    def set_tilt(self, tilt: float) -> None:
        self._tilt = tilt

    def set_ruled_twist(self) -> None:
        shpa = self.scta.get_shape()
        shpb = self.sctb.get_shape()
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
            self._tilt = self.sctb.tilt + self.bval*(self.sctb.tilt - self.scta.tilt)
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

    def get_profile(self, offset: bool=True) -> 'MatrixVector':
        prfa = self.scta.get_profile(offset=offset)
        prfb = self.sctb.get_profile(offset=offset)
        profiledir = prfb - prfa
        return prfa + self.bval*profiledir

    def get_shape(self) -> 'MatrixVector':
        profile = self.get_profile()
        scaledprofile = profile*self.chord
        rotatedprofile = vector_to_global(self.crdsys, scaledprofile)
        translatedprofile = rotatedprofile + self.point
        return translatedprofile

    def mesh_grids(self, gid: int) -> int:
        shp = self.get_shape()
        num = shp.shape[1]
        self.grds = []
        for i in range(num):
            te = False
            if i == 0 or i == num-1:
                if not self.nohsv:
                    te = True
            self.grds.append(Grid(gid, shp[0, i].x, shp[0, i].y, shp[0, i].z, te))
            gid += 1
        return gid

    def __repr__(self) -> str:
        return f'<PanelProfile at {self.point:}>'
