from pygeom.geom3d import Vector
from pygeom.matrix3d import MatrixVector
from numpy.matlib import matrix, where

class Grid(Vector):
    gid: None
    te: None
    def __init__(self, gid: int, x: float, y: float, z: float, te: bool):
        self.gid = gid
        super().__init__(x, y, z)
        self.te = te
    def __str__(self):
        outstr = '{:}: <{:}, {:}, {:}>'.format(self.gid, self.x, self.y, self.z)
        if self.te:
            outstr += ', Trailing Edge: True'
        return outstr
    def __format__(self, format_spec):
        frmstr = '{:}: <{:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'}>'
        outstr = frmstr.format(self.gid, self.x, self.y, self.z)
        if self.te:
            outstr += ', Trailing Edge: True'
        return outstr
