from pygeom.geom3d import Vector
from pygeom.matrix3d import MatrixVector
from pygeom.geom2d import Vector2D
from numpy.matlib import zeros
from .triangle import TriangleEdge
from .edgecalculator import EdgeCalculator
from .trailingconverter import TrailingConverter#, BoundConverter
from .triangleconverter import TriangleConverter

class BoundEdge(TriangleEdge):
    dirl: Vector = None
    def __init__(self, grda: Vector, grdb: Vector, dirz: Vector, dirl: Vector):
        super().__init__(grda, grdb, dirz)
        self.dirl = dirl
    def linear_doublet_phi(self, rela: MatrixVector, relb: MatrixVector):
        x = rela*self.dirx
        ya = rela*self.diry
        yb = relb*self.diry
        z = relb*self.dirz
        calc = EdgeCalculator(x, ya, yb, z)
        relc = relb + self.lenab*self.dirl
        xc = relc*self.dirx
        yc = relc*self.diry
        triaconv = TriangleConverter(x, ya, x, yb, xc, yc)
        # dx = -self.dirx*self.dirl
        # dy = -self.diry*self.dirl
        # triaconv = BoundConverter(x, ya, x, yb, dx, dy)
        phidabc = triaconv.return_linear(calc.phido, calc.phidx, calc.phidy)
        return phidabc[0], phidabc[1] + phidabc[2]
    def quadratic_doublet_phi(self, rela: MatrixVector, relb: MatrixVector):
        x = rela*self.dirx
        ya = rela*self.diry
        yb = relb*self.diry
        z = relb*self.dirz
        calc = EdgeCalculator(x, ya, yb, z)
        relc = relb + self.lenab*self.dirl
        xc = relc*self.dirx
        yc = relc*self.diry
        triaconv = TriangleConverter(x, ya, x, yb, xc, yc)
        # dx = -self.dirx*self.dirl
        # dy = -self.diry*self.dirl
        # triaconv = BoundConverter(x, ya, x, yb, dx, dy)
        phidabc = triaconv.return_quadratic(calc.phido, calc.phidx, calc.phidy,
                                            calc.phidxx, calc.phidxy, calc.phidyy)
        return phidabc[0], phidabc[1] + phidabc[2] + phidabc[4], phidabc[3] + phidabc[5]

class TrailingEdgeA():
    grdo: Vector = None
    dirz: Vector = None
    _diry: Vector = None
    _dirx: Vector = None
    def __init__(self, grdo: Vector, dirl: Vector, dirz: Vector):
        self.grdo = grdo
        self.dirl = dirl.to_unit()
        self.dirz = dirz.to_unit()
    @property
    def diry(self):
        if self._diry is None:
            self._diry = -self.dirl
        return self._diry
    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = (self.dirz**self.diry).to_unit()
        return self._dirx
    def constant_doublet_phi(self, relo: MatrixVector):
        xo = relo*self.dirx
        yo = relo*self.diry
        zo = relo*self.dirz
        calc = EdgeCalculator(xo, yo, float('inf'), zo)
        return calc.phido
    def linear_doublet_phi(self, relo: MatrixVector, relc: MatrixVector):
        xo = relo*self.dirx
        yo = relo*self.diry
        zo = relo*self.dirz
        calc = EdgeCalculator(xo, yo, float('inf'), zo)
        xc = relc*self.dirx
        triaconv = TrailingConverter(xo, xc)
        phidab = triaconv.return_linear(calc.phido, calc.phidx)
        return phidab
    def quadratic_doublet_phi(self, relo: MatrixVector, relc: MatrixVector):
        xo = relo*self.dirx
        yo = relo*self.diry
        zo = relo*self.dirz
        calc = EdgeCalculator(xo, yo, float('inf'), zo)
        xc = relc*self.dirx
        yc = relc*self.diry
        xb = xo
        yb = yo+1.0
        triaconv = TriangleConverter(xo, yo, xb, yb, xc, yc)
        phidabc = triaconv.return_quadratic(calc.phido, calc.phidx, calc.phidy,
                                            calc.phidxx, calc.phidxy, calc.phidyy)
        phida = phidabc[0] + phidabc[1] + phidabc[3]
        phidab = phidabc[4] + phidabc[5]
        phidb = phidabc[2]
        # triaconv = TrailingConverter(xo, xc)
        # phidabc = triaconv.return_quadratic(calc.phido, calc.phidx, calc.phidxx)
        # return phidabcs
        return phida, phidb, phidab

class TrailingEdgeB():
    grdo: Vector = None
    dirz: Vector = None
    _diry: Vector = None
    _dirx: Vector = None
    def __init__(self, grdo: Vector, dirl: Vector, dirz: Vector):
        self.grdo = grdo
        self.dirl = dirl.to_unit()
        self.dirz = dirz.to_unit()
    @property
    def diry(self):
        if self._diry is None:
            self._diry = self.dirl
        return self._diry
    @property
    def dirx(self):
        if self._dirx is None:
            self._dirx = (self.dirz**self.diry).to_unit()
        return self._dirx
    def constant_doublet_phi(self, relo: MatrixVector):
        xo = relo*self.dirx
        yo = relo*self.diry
        zo = relo*self.dirz
        calc = EdgeCalculator(xo, float('-inf'), yo, zo)
        return calc.phido
    def linear_doublet_phi(self, relo: MatrixVector, relc: MatrixVector):
        xo = relo*self.dirx
        yo = relo*self.diry
        zo = relo*self.dirz
        calc = EdgeCalculator(xo, float('-inf'), yo, zo)
        xc = relc*self.dirx
        triaconv = TrailingConverter(xo, xc)
        phidab = triaconv.return_linear(calc.phido, calc.phidx)
        return phidab
    def quadratic_doublet_phi(self, relo: MatrixVector, relc: MatrixVector):
        xo = relo*self.dirx
        yo = relo*self.diry
        zo = relo*self.dirz
        calc = EdgeCalculator(xo, float('-inf'), yo, zo)
        xc = relc*self.dirx
        yc = relc*self.diry
        xb = xo
        yb = yo+1.0
        triaconv = TriangleConverter(xo, yo, xb, yb, xc, yc)
        phidabc = triaconv.return_quadratic(calc.phido, calc.phidx, calc.phidy,
                                            calc.phidxx, calc.phidxy, calc.phidyy)
        phida = phidabc[0] + phidabc[1] + phidabc[3]
        phidab = phidabc[4] + phidabc[5]
        phidb = phidabc[2]
        # triaconv = TrailingConverter(xo, xc)
        # phidabc = triaconv.return_quadratic(calc.phido, calc.phidx, calc.phidxx)
        # return phidabc
        return phida, phidb, phidab

class TrailingDoublet():
    grda: Vector = None
    grdb: Vector = None
    dirl: Vector = None
    chka: bool = None
    chkb: bool = None
    chkc: bool = None
    _vecs: Vector = None
    _lens: float = None
    _dirs: Vector = None
    _dirx: Vector = None
    _diry: Vector = None
    _dirz: Vector = None
    _edga: TrailingEdgeA = None
    _edgb: BoundEdge = None
    _edgc: TrailingEdgeB = None
    _pnta: Vector2D = None
    _pntb: Vector2D = None
    _pntc: Vector2D = None
    _jac: float = None
    _area: float = None
    def __init__(self, grda: Vector, grdb: Vector, dirl: Vector):
        self.grda = grda
        self.grdb = grdb
        self.dirl = dirl.to_unit()
        self.chka = True
        self.chkb = True
        self.chkc = True
    @property
    def vecs(self):
        if self._vecs is None:
            self._vecs = self.grdb-self.grda
        return self._vecs
    @property
    def lens(self):
        if self._lens is None:
            self._lens = self.vecs.return_magnitude()
        return self._lens
    @property
    def dirs(self):
        if self._dirs is None:
            self._dirs = self.vecs/self.lens
        return self._dirs
    @property
    def dirz(self):
        if self._dirz is None:
            self._dirz = (self.dirs**self.dirl).to_unit()
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
    @property
    def edga(self):
        if self._edga is None:
            self._edga = TrailingEdgeA(self.grda, self.dirl, self.dirz)
        return self._edga
    @property
    def edgb(self):
        if self._edgb is None:
            self._edgb = BoundEdge(self.grda, self.grdb, self.dirz, self.dirl)
        return self._edgb
    @property
    def edgc(self):
        if self._edgc is None:
            self._edgc = TrailingEdgeB(self.grdb, self.dirl, self.dirz)
        return self._edgc
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
    def mach_correction(self, matvec: MatrixVector,
                        betx: float=1.0, bety: float=1.0, betz: float=1.0):
        if betx != 1.0:
            matvec.x = matvec.x/betx
        if bety != 1.0:
            matvec.y = matvec.y/bety
        if betz != 1.0:
            matvec.z = matvec.z/betz
        return matvec
    def constant_doublet_phi(self, pnts: MatrixVector,
                             betx: float=1.0, bety: float=1.0, betz: float=1.0):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        rela = self.mach_correction(rela, betx, bety, betz)
        relb = self.mach_correction(relb, betx, bety, betz)
        phido = zeros(pnts.shape, dtype=float)
        if self.chka:
            phidoa = self.edga.constant_doublet_phi(rela)
            phido += phidoa
        if self.chkb:
            phidob = self.edgb.constant_doublet_phi(rela, relb)
            phido += phidob
        if self.chkc:
            phidoc = self.edgc.constant_doublet_phi(relb)
            phido += phidoc
        return phido
    def linear_doublet_phi(self, pnts: MatrixVector,
                           betx: float=1.0, bety: float=1.0, betz: float=1.0):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        rela = self.mach_correction(rela, betx, bety, betz)
        relb = self.mach_correction(relb, betx, bety, betz)
        phida = zeros(pnts.shape, dtype=float)
        phidb = zeros(pnts.shape, dtype=float)
        if self.chka:
            phidaab = self.edga.linear_doublet_phi(rela, relb)
            phida += phidaab[0]
            phidb += phidaab[1]
        if self.chkb:
            phidbab = self.edgb.linear_doublet_phi(rela, relb)
            phida += phidbab[0]
            phidb += phidbab[1]
        if self.chkc:
            phidcab = self.edgc.linear_doublet_phi(relb, rela)
            phida += phidcab[1]
            phidb += phidcab[0]
        return phida, phidb
    def quadratic_doublet_phi(self, pnts: MatrixVector,
                              betx: float=1.0, bety: float=1.0, betz: float=1.0):
        rela = pnts-self.grda
        relb = pnts-self.grdb
        rela = self.mach_correction(rela, betx, bety, betz)
        relb = self.mach_correction(relb, betx, bety, betz)
        phida = zeros(pnts.shape, dtype=float)
        phidb = zeros(pnts.shape, dtype=float)
        phidab = zeros(pnts.shape, dtype=float)
        if self.chka:
            phidaab = self.edga.quadratic_doublet_phi(rela, relb)
            phida += phidaab[0]
            phidb += phidaab[1]
            phidab += phidaab[2]
        if self.chkb:
            phidbab = self.edgb.quadratic_doublet_phi(rela, relb)
            phida += phidbab[0]
            phidb += phidbab[1]
            phidab += phidbab[2]
        if self.chkc:
            phidcab = self.edgc.quadratic_doublet_phi(relb, rela)
            phida += phidcab[1]
            phidb += phidcab[0]
            phidab += phidcab[2]
        return phida, phidb, phidab
