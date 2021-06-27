from typing import List
from pygeom.geom3d import Vector
from pygeom.matrix3d import MatrixVector
from numpy.matlib import zeros
from .triangle import Triangle

class Quadrilateral():
    grda: Vector = None
    grdb: Vector = None
    grdc: Vector = None
    grdd: Vector = None
    dirl: Vector = None
    chka: bool = None
    chkb: bool = None
    chkc: bool = None
    chkd: bool = None
    _grds: List[Vector] = None
    _grde: Vector = None
    _trias: List[Triangle] = None
    _area: float = None
    def __init__(self, grda: Vector, grdb: Vector,
                 grdc: Vector, grdd: Vector, dirl: Vector):
        self.grda = grda
        self.grdb = grdb
        self.grdc = grdc
        self.grdd = grdd
        self.dirl = dirl
        self.chka = True
        self.chkb = True
        self.chkc = True
        self.chkd = True
    @property
    def grds(self) -> List[Vector]:
        if self._grds is None:
            self._grds = [self.grda, self.grdb, self.grdc, self.grdd]
        return self._grds
    @property
    def grde(self) -> Vector:
        if self._grde is None:
            self._grde = sum(self.grds)/4
        return self._grde
    @property
    def trias(self) -> List[Triangle]:
        if self._trias is None:
            self._trias = []
            self._trias.append(Triangle(self.grda, self.grdb, self.grde, self.dirl))
            self._trias.append(Triangle(self.grdb, self.grdc, self.grde, self.dirl))
            self._trias.append(Triangle(self.grdc, self.grdd, self.grde, self.dirl))
            self._trias.append(Triangle(self.grdd, self.grda, self.grde, self.dirl))
        return self._trias
    @property
    def area(self):
        if self._area is None:
            self._area = sum([tria.area for tria in self.trias])
        return self._area
    def constant_phi(self, pnts: MatrixVector):
        phido = zeros(pnts.shape, dtype=float)
        phiso = zeros(pnts.shape, dtype=float)
        chk = True
        tria = self.trias[0]
        if not self.chka:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidoa, phisoa = tria.constant_phi(pnts)
        if not self.chka:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phiso += phisoa
        phido += phidoa
        tria = self.trias[1]
        if not self.chkb:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidob, phisob = tria.constant_phi(pnts)
        if not self.chkb:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phiso += phisob
        phido += phidob
        tria = self.trias[2]
        if not self.chkc:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidoc, phisoc = tria.constant_phi(pnts)
        if not self.chkc:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phiso += phisoc
        phido += phidoc
        tria = self.trias[3]
        if not self.chkd:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidoc, phisoc = tria.constant_phi(pnts)
        if not self.chkd:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phiso += phisoc
        phido += phidoc
        return phido, phiso
    def linear_phi(self, pnts: MatrixVector):
        phida = zeros(pnts.shape, dtype=float)
        phidb = zeros(pnts.shape, dtype=float)
        phidc = zeros(pnts.shape, dtype=float)
        phidd = zeros(pnts.shape, dtype=float)
        phide = zeros(pnts.shape, dtype=float)
        phisa = zeros(pnts.shape, dtype=float)
        phisb = zeros(pnts.shape, dtype=float)
        phisc = zeros(pnts.shape, dtype=float)
        phisd = zeros(pnts.shape, dtype=float)
        phise = zeros(pnts.shape, dtype=float)
        chk = True
        tria = self.trias[0]
        if not self.chka:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidabe, phisabe = tria.linear_phi(pnts)
        if not self.chka:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phida += phidabe[0]
        phidb += phidabe[1]
        phide += phidabe[2]
        phisa += phisabe[0]
        phisb += phisabe[1]
        phise += phisabe[2]
        tria = self.trias[1]
        if not self.chkb:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidbce, phisbce = tria.linear_phi(pnts)
        if not self.chkb:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phidb += phidbce[0]
        phidc += phidbce[1]
        phide += phidbce[2]
        phisb += phisbce[0]
        phisc += phisbce[1]
        phise += phisbce[2]
        tria = self.trias[2]
        if not self.chkc:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phidcde, phiscde = tria.linear_phi(pnts)
        if not self.chkc:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phidc += phidcde[0]
        phidd += phidcde[1]
        phide += phidcde[2]
        phisc += phiscde[0]
        phisd += phiscde[1]
        phise += phiscde[2]
        tria = self.trias[3]
        if not self.chkd:
            tria.chka = False
            tria.chkb = chk
            tria.chkc = chk
        phiddae, phisdae = tria.linear_phi(pnts)
        if not self.chkd:
            tria.chka = True
            tria.chkb = True
            tria.chkc = True
        phidd += phiddae[0]
        phida += phiddae[1]
        phide += phiddae[2]
        phisd += phisdae[0]
        phisa += phisdae[1]
        phise += phisdae[2]
        phideo4 = phide/4
        phiseo4 = phise/4
        phida += phideo4
        phidb += phideo4
        phidc += phideo4
        phidd += phideo4
        phisa += phiseo4
        phisb += phiseo4
        phisc += phiseo4
        phisd += phiseo4
        return (phida, phidb, phidc, phidd), (phisa, phisb, phisc, phisd)
