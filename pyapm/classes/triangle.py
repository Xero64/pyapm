from typing import List
from math import sqrt
from pygeom.geom3d import Vector, ihat, khat
from pygeom.geom2d import Vector2D
from pygeom.matrix3d import MatrixVector
from numpy.matlib import zeros
from .grid import Grid
from .edgecalculator import EdgeCalculator
from .triangleconverter import TriangleConverter

class TriangleEdge():
    grda: Grid = None
    grdb: Grid = None
    dirz: Vector = None
    _vecab: Vector = None
    _lenab: float = None
    _dirab: Vector = None
    _dirx: Vector = None
    def __init__(self, grda: Grid, grdb: Grid, dirz: Vector):
        self.grda = grda
        self.grdb = grdb
        self.dirz = dirz.to_unit()
    @property
    def vecab(self):
        if self._vecab is None:
            self._vecab = self.grdb-self.grda
        return self._vecab
    @property
    def lenab(self):
        if self._lenab is None:
            self._lenab = self.vecab.return_magnitude()
        return self._lenab
    @property
    def dirab(self):
        if self._dirab is None:
            self._dirab = self.vecab/self.lenab
        return self._dirab
    @property
    def diry(self):
        return self.dirab
    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = (self.diry**self.dirz).to_unit()
        return self._dirx
    def constant_phi_pnts(self, pnts: MatrixVector):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        return self.constant_phi(rela, relb)
    def constant_phi(self, rela: MatrixVector, relb: MatrixVector):
        x = rela*self.dirx
        ya = rela*self.diry
        yb = relb*self.diry
        z = relb*self.dirz
        calc = EdgeCalculator(x, ya, yb, z)
        return calc.phido, calc.phiso
    def linear_phi_pnts(self, pnts: MatrixVector):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        relc = (rela+relb)/2 + self.dirx*self.lenab
        phidabc, phisabc = self.linear_phi(rela, relb, relc)
        phida = phidabc[0]+phidabc[2]/2
        phidb = phidabc[1]+phidabc[2]/2
        phisa = phisabc[0]+phisabc[2]/2
        phisb = phisabc[1]+phisabc[2]/2
        return (phida, phidb), (phisa, phisb)
    def linear_phi(self, rela: MatrixVector, relb: MatrixVector, relc: MatrixVector):
        x = rela*self.dirx
        ya = rela*self.diry
        yb = relb*self.diry
        z = relb*self.dirz
        calc = EdgeCalculator(x, ya, yb, z)
        xc = relc*self.dirx
        yc = relc*self.diry
        triaconv = TriangleConverter(x, ya, x, yb, xc, yc)
        phidabc = triaconv.return_linear(calc.phido, calc.phidx, calc.phidy)
        phisabc = triaconv.return_linear(calc.phiso, calc.phisx, calc.phisy)
        return phidabc, phisabc
    def quadratic_phi(self, rela: MatrixVector, relb: MatrixVector, relc: MatrixVector):
        x = rela*self.dirx
        ya = rela*self.diry
        yb = relb*self.diry
        z = relb*self.dirz
        calc = EdgeCalculator(x, ya, yb, z)
        xc = relc*self.dirx
        yc = relc*self.diry
        triaconv = TriangleConverter(x, ya, x, yb, xc, yc)
        phidabc = triaconv.return_quadratic(calc.phido, calc.phidx, calc.phidy,
                                            calc.phidxx, calc.phidxy, calc.phidyy)
        phisabc = triaconv.return_linear(calc.phiso, calc.phisx, calc.phisy)
        return phidabc, phisabc
    def constant_doublet_phi(self, rela: MatrixVector, relb: MatrixVector):
        x = rela*self.dirx
        ya = rela*self.diry
        yb = relb*self.diry
        z = relb*self.dirz
        calc = EdgeCalculator(x, ya, yb, z)
        return calc.phido
    def __repr__(self):
        if hasattr(self.grda, 'gid') and hasattr(self.grdb, 'gid'):
            return f'<pyapm.TriangleEdge {self.grda.gid:d} - {self.grdb.gid:d}>'
        else:
            return '<pyapm.TriangleEdge>'

oor2 = 1/sqrt(2.0)

class Triangle():
    grda: Grid = None
    grdb: Grid = None
    grdc: Grid = None
    chka: bool = None
    chkb: bool = None
    chkc: bool = None
    _dirl: Vector = None
    _grds: List[Grid] = None
    _dirx: Vector = None
    _diry: Vector = None
    _dirz: Vector = None
    _edga: TriangleEdge = None
    _edgb: TriangleEdge = None
    _edgc: TriangleEdge = None
    _edgs: List[TriangleEdge] = None
    _pnta: Vector2D = None
    _pntb: Vector2D = None
    _pntc: Vector2D = None
    _jac: float = None
    _area: float = None
    def __init__(self, grda: Grid, grdb: Grid, grdc: Grid):
        self.grda = grda
        self.grdb = grdb
        self.grdc = grdc
        self.chka = True
        self.chkb = True
        self.chkc = True
    @property
    def grds(self):
        if self._grds is None:
            self._grds = [self.grda, self.grdb, self.grdc]
        return self._grds
    @property
    def dirz(self):
        if self._dirz is None:
            vecab = self.grdb-self.grda
            vecbc = self.grdc-self.grdb
            self._dirz = (vecab**vecbc).to_unit()
        return self._dirz
    # @property
    # def dirl(self):
    #     if self._dirl is None:
    #         self._dirl = (self.grdb-self.grda).to_unit()
    #     return self._dirl
    @property
    def diry(self):
        if self._diry is None:
            vecy = self.dirz**ihat
            magy = vecy.return_magnitude()
            if magy < oor2:
                vecy = self.dirz**khat
            self._diry = vecy.to_unit()
        return self._diry
    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = (self.diry**self.dirz).to_unit()
        return self._dirx
    @property
    def edga(self):
        if self._edga is None:
            self._edga = TriangleEdge(self.grda, self.grdb, self.dirz)
        return self._edga
    @property
    def edgb(self):
        if self._edgb is None:
            self._edgb = TriangleEdge(self.grdb, self.grdc, self.dirz)
        return self._edgb
    @property
    def edgc(self):
        if self._edgc is None:
            self._edgc = TriangleEdge(self.grdc, self.grda, self.dirz)
        return self._edgc
    @property
    def edgs(self):
        if self._edgs is None:
            self._edgs = [self.edga, self.edgb, self.edgc]
        return self._edgs
    @property
    def pnta(self):
        if self._pnta is None:
            self._pnta = Vector2D(self.grda*self.dirx, self.grda*self.diry)
        return self._pnta
    @property
    def pntb(self):
        if self._pntb is None:
            self._pntb = Vector2D(self.grdb*self.dirx, self.grdb*self.diry)
        return self._pntb
    @property
    def pntc(self):
        if self._pntc is None:
            self._pntc = Vector2D(self.grdc*self.dirx, self.grdc*self.diry)
        return self._pntc
    @property
    def jac(self):
        if self._jac is None:
            self._jac = 0.0
            self._jac += self.pnta.x*self.pntb.y
            self._jac -= self.pnta.x*self.pntc.y
            self._jac -= self.pntb.x*self.pnta.y
            self._jac += self.pntb.x*self.pntc.y
            self._jac += self.pntc.x*self.pnta.y
            self._jac -= self.pntc.x*self.pntb.y
        return self._jac
    @property
    def area(self):
        if self._area is None:
            self._area = self.jac/2
        return self._area
    def mach_correction(self, matvec: MatrixVector,
                        betx: float=1.0, bety: float=1.0, betz: float=1.0):
        if betx != 1.0:
            matvec.x = matvec.x/betx
        if bety != 1.0:
            matvec.y = matvec.y/bety
        if betz != 1.0:
            matvec.z = matvec.z/betz
        return matvec
    def constant_phi(self, pnts: MatrixVector,
                     betx: float=1.0, bety: float=1.0, betz: float=1.0):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        relc = pnts-self.grdc
        rela = self.mach_correction(rela, betx, bety, betz)
        relb = self.mach_correction(relb, betx, bety, betz)
        relc = self.mach_correction(relc, betx, bety, betz)
        phido = zeros(pnts.shape, dtype=float)
        phiso = zeros(pnts.shape, dtype=float)
        if self.chka:
            phidoa, phisoa = self.edga.constant_phi(rela, relb)
            phiso += phisoa
            phido += phidoa
        if self.chkb:
            phidob, phisob = self.edgb.constant_phi(relb, relc)
            phiso += phisob
            phido += phidob
        if self.chkc:
            phidoc, phisoc = self.edgc.constant_phi(relc, rela)
            phiso += phisoc
            phido += phidoc
        return phido, phiso
    def linear_phi(self, pnts: MatrixVector,
                   betx: float=1.0, bety: float=1.0, betz: float=1.0):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        relc = pnts-self.grdc
        rela = self.mach_correction(rela, betx, bety, betz)
        relb = self.mach_correction(relb, betx, bety, betz)
        relc = self.mach_correction(relc, betx, bety, betz)
        phida = zeros(pnts.shape, dtype=float)
        phidb = zeros(pnts.shape, dtype=float)
        phidc = zeros(pnts.shape, dtype=float)
        phisa = zeros(pnts.shape, dtype=float)
        phisb = zeros(pnts.shape, dtype=float)
        phisc = zeros(pnts.shape, dtype=float)
        if self.chka:
            phidaabc, phisaabc = self.edga.linear_phi(rela, relb, relc)
            phida += phidaabc[0]
            phidb += phidaabc[1]
            phidc += phidaabc[2]
            phisa += phisaabc[0]
            phisb += phisaabc[1]
            phisc += phisaabc[2]
        if self.chkb:
            phidbabc, phisbabc = self.edgb.linear_phi(relb, relc, rela)
            phidb += phidbabc[0]
            phidc += phidbabc[1]
            phida += phidbabc[2]
            phisb += phisbabc[0]
            phisc += phisbabc[1]
            phisa += phisbabc[2]
        if self.chkc:
            phidcabc, phiscabc = self.edgc.linear_phi(relc, rela, relb)
            phidc += phidcabc[0]
            phida += phidcabc[1]
            phidb += phidcabc[2]
            phisc += phiscabc[0]
            phisa += phiscabc[1]
            phisb += phiscabc[2]
        return (phida, phidb, phidc), (phisa, phisb, phisc)
    def quadratic_phi(self, pnts: MatrixVector,
                      betx: float=1.0, bety: float=1.0, betz: float=1.0):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        relc = pnts-self.grdc
        rela = self.mach_correction(rela, betx, bety, betz)
        relb = self.mach_correction(relb, betx, bety, betz)
        relc = self.mach_correction(relc, betx, bety, betz)
        phida = zeros(pnts.shape, dtype=float)
        phidb = zeros(pnts.shape, dtype=float)
        phidc = zeros(pnts.shape, dtype=float)
        phidab = zeros(pnts.shape, dtype=float)
        phidbc = zeros(pnts.shape, dtype=float)
        phidca = zeros(pnts.shape, dtype=float)
        phisa = zeros(pnts.shape, dtype=float)
        phisb = zeros(pnts.shape, dtype=float)
        phisc = zeros(pnts.shape, dtype=float)
        if self.chka:
            phidaabc, phisaabc = self.edga.quadratic_phi(rela, relb, relc)
            phida += phidaabc[0]
            phidb += phidaabc[1]
            phidc += phidaabc[2]
            phidab += phidaabc[3]
            phidbc += phidaabc[4]
            phidca += phidaabc[5]
            phisa += phisaabc[0]
            phisb += phisaabc[1]
            phisc += phisaabc[2]
        if self.chkb:
            phidbabc, phisbabc = self.edgb.quadratic_phi(relb, relc, rela)
            phidb += phidbabc[0]
            phidc += phidbabc[1]
            phida += phidbabc[2]
            phidbc += phidbabc[3]
            phidca += phidbabc[4]
            phidab += phidbabc[5]
            phisb += phisbabc[0]
            phisc += phisbabc[1]
            phisa += phisbabc[2]
        if self.chkc:
            phidcabc, phiscabc = self.edgc.quadratic_phi(relc, rela, relb)
            phidc += phidcabc[0]
            phida += phidcabc[1]
            phidb += phidcabc[2]
            phidca += phidcabc[3]
            phidab += phidcabc[4]
            phidbc += phidcabc[5]
            phisc += phiscabc[0]
            phisa += phiscabc[1]
            phisb += phiscabc[2]
        return (phida, phidb, phidc, phidab, phidbc, phidca), (phisa, phisb, phisc)
    def __repr__(self) -> str:
        if hasattr(self.grda, 'gid') and hasattr(self.grdb, 'gid') and hasattr(self.grdc, 'gid'):
            return f'<pyapm.Triangle {self.grda.gid:d}, {self.grdb.gid:d}, {self.grdc.gid:d}>'
        else:
            return '<pyapm.Triangle>'
