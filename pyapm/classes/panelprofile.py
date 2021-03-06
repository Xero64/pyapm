from math import radians, sin, cos, acos, degrees
from typing import List
from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import vector_to_global
from .grid import Grid

class PanelProfile(object):
    point: Vector = None
    chord: float = None
    twist: float = None
    _tilt: float = None
    _crdsys: Coordinate = None
    bval: float = None
    bpos: float = None
    scta: object = None
    sctb: object = None
    grds: List[Grid] = None
    nohsv: bool = None
    def __init__(self, point: Vector, chord: float, twist: float):
        self.point = point
        self.chord = chord
        self.twist = twist
    def set_tilt(self, tilt: float):
        self._tilt = tilt
    def set_ruled_twist(self):
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
    def tilt(self):
        if self._tilt is None:
            self._tilt = self.sctb.tilt + self.bval*(self.sctb.tilt - self.scta.tilt)
        return self._tilt
    @property
    def crdsys(self):
        if self._crdsys is None:
            tilt = radians(self.tilt)
            sintilt = sin(tilt)
            costilt = cos(tilt)
            diry = Vector(0.0, costilt, sintilt)
            twist = radians(self.twist)
            sintwist = sin(twist)
            costwist = cos(twist)
            dirx = Vector(costwist, sintwist*sintilt, -sintwist*costilt)
            dirz = dirx**diry
            self._crdsys = Coordinate(self.point, dirx, diry, dirz)
        return self._crdsys
    def get_profile(self, offset: bool=True):
        prfa = self.scta.get_profile(offset=offset)
        prfb = self.sctb.get_profile(offset=offset)
        profiledir = prfb - prfa
        return prfa + self.bval*profiledir
    def get_shape(self):
        profile = self.get_profile()
        scaledprofile = profile*self.chord
        rotatedprofile = vector_to_global(self.crdsys, scaledprofile)
        translatedprofile = rotatedprofile + self.point
        return translatedprofile
    def mesh_grids(self, gid: int):
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
    def __repr__(self):
        return f'<PanelProfile at {self.point:}>'
