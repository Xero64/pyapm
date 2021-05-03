from pygeom import Vector

class Triangle(object):
    grda: Vector = None
    grdb: Vector = None
    grdc: Vector = None
    dirl: Vector = None
    _dirx: Vector = None
    _diry: Vector = None
    _dirz: Vector = None
    def __init__(self, grda: Vector, grdb: Vector, grdc: Vector, dirl: Vector):
        self.grda = grda
        self.grdb = grdb
        self.grdc = grdc
        self.dirl = dirl.to_unit()
    @property
    def dirz(self):
        if self._dirz is None:
            vecab = self.grdb-self.grda
            vecbc = self.grdc-self.grdb
            self._dirz = (vecab**vecbc).to_unit()
        return self._dirz
    @property
    def diry(self):
        if self._diry is None:
            self._diry = (self.dirz**self.dirl).to_unit()
        return self._diry
    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = (self.diry**self.dirz).to_unit()
        return self._dirx
