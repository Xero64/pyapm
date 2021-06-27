from typing import Tuple
from numpy import multiply, square, divide
from numpy.matlib import matrix

class TriangleConverter():
    xa: matrix = None
    ya: matrix = None
    xb: matrix = None
    yb: matrix = None
    xc: matrix = None
    yc: matrix = None
    _jac: matrix = None
    _jac2: float = None
    _xba: matrix = None
    _yab: matrix = None
    _xcb: matrix = None
    _ybc: matrix = None
    _xac: matrix = None
    _yca: matrix = None
    _xyab: matrix = None
    _xybc: matrix = None
    _xyca: matrix = None
    _xbaoj: matrix = None
    _yaboj: matrix = None
    _xcboj: matrix = None
    _ybcoj: matrix = None
    _xacoj: matrix = None
    _ycaoj: matrix = None
    _xyaboj: matrix = None
    _xybcoj: matrix = None
    _xycaoj: matrix = None
    _qa: Tuple[matrix] = None
    _qb: Tuple[matrix] = None
    _qc: Tuple[matrix] = None
    _qab: Tuple[matrix] = None
    _qbc: Tuple[matrix] = None
    _qca: Tuple[matrix] = None
    def __init__(self, xa: matrix, ya: matrix, xb: matrix, yb: matrix,
                 xc: matrix, yc: matrix):
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        self.xc = xc
        self.yc = yc
    def set_jac(self, jac: float):
        self._jac = jac
    @property
    def xba(self) -> matrix:
        if self._xba is None:
            self._xba = self.xb - self.xa
        return self._xba
    @property
    def yab(self) -> matrix:
        if self._yab is None:
            self._yab = self.ya - self.yb
        return self._yab
    @property
    def xcb(self) -> matrix:
        if self._xcb is None:
            self._xcb = self.xc - self.xb
        return self._xcb
    @property
    def ybc(self) -> matrix:
        if self._ybc is None:
            self._ybc = self.yb - self.yc
        return self._ybc
    @property
    def xac(self) -> matrix:
        if self._xac is None:
            self._xac = self.xa - self.xc
        return self._xac
    @property
    def yca(self) -> matrix:
        if self._yca is None:
            self._yca = self.yc - self.ya
        return self._yca
    @property
    def xyab(self) -> matrix:
        if self._xyab is None:
            self._xyab = multiply(self.xa, self.yb) - multiply(self.xb, self.ya)
        return self._xyab
    @property
    def xybc(self) -> matrix:
        if self._xybc is None:
            self._xybc = multiply(self.xb, self.yc) - multiply(self.xc, self.yb)
        return self._xybc
    @property
    def xyca(self) -> matrix:
        if self._xyca is None:
            self._xyca = multiply(self.xc, self.ya) - multiply(self.xa, self.yc)
        return self._xyca
    @property
    def jac(self) -> matrix:
        if self._jac is None:
            self._jac = self.xyab + self.xybc + self.xyca
        return self._jac
    @property
    def jac2(self) -> matrix:
        if self._jac2 is None:
            self._jac2 = square(self.jac)
        return self._jac2
    @property
    def xbaoj(self) -> matrix:
        if self._xbaoj is None:
            self._xbaoj = divide(self.xba, self.jac)
        return self._xbaoj
    @property
    def yaboj(self) -> matrix:
        if self._yaboj is None:
            self._yaboj = divide(self.yab, self.jac)
        return self._yaboj
    @property
    def xcboj(self) -> matrix:
        if self._xcboj is None:
            self._xcboj = divide(self.xcb, self.jac)
        return self._xcboj
    @property
    def ybcoj(self) -> matrix:
        if self._ybcoj is None:
            self._ybcoj = divide(self.ybc, self.jac)
        return self._ybcoj
    @property
    def xacoj(self) -> matrix:
        if self._xacoj is None:
            self._xacoj = divide(self.xac, self.jac)
        return self._xacoj
    @property
    def ycaoj(self) -> matrix:
        if self._ycaoj is None:
            self._ycaoj = divide(self.yca, self.jac)
        return self._ycaoj
    @property
    def xyaboj(self) -> matrix:
        if self._xyaboj is None:
            self._xyaboj = divide(self.xyab, self.jac)
        return self._xyaboj
    @property
    def xybcoj(self) -> matrix:
        if self._xybcoj is None:
            self._xybcoj = divide(self.xybc, self.jac)
        return self._xybcoj
    @property
    def xycaoj(self) -> matrix:
        if self._xycaoj is None:
            self._xycaoj = divide(self.xyca, self.jac)
        return self._xycaoj
    def return_linear(self, valo: matrix, valx: matrix, valy: matrix):
        vala = multiply(self.xybcoj, valo)
        vala += multiply(self.ybcoj, valx)
        vala += multiply(self.xcboj, valy)
        valb = multiply(self.xycaoj, valo)
        valb += multiply(self.ycaoj, valx)
        valb += multiply(self.xacoj, valy)
        valc = multiply(self.xyaboj, valo)
        valc += multiply(self.yaboj, valx)
        valc += multiply(self.xbaoj, valy)
        return vala, valb, valc
    @property
    def qa(self) -> Tuple[matrix]:
        if self._qa is None:
            self._qa = (
                multiply(self.xybcoj, 2*self.xybcoj - 1.0),
                multiply(self.ybcoj, 4*self.xybcoj - 1.0),
                multiply(self.xcboj, 4*self.xybcoj - 1.0),
                2*square(self.ybcoj),
                4*multiply(self.xcboj, self.ybcoj),
                2*square(self.xcboj)
            )
        return self._qa
    @property
    def qb(self) -> Tuple[matrix]:
        if self._qb is None:
            self._qb = (
                multiply(self.xycaoj, 2*self.xycaoj - 1.0),
                multiply(self.ycaoj, 4*self.xycaoj - 1.0),
                multiply(self.xacoj, 4*self.xycaoj - 1.0),
                2*square(self.ycaoj),
                4*multiply(self.xacoj, self.ycaoj),
                2*square(self.xacoj)
            )
        return self._qb
    @property
    def qc(self) -> Tuple[matrix]:
        if self._qc is None:
            self._qc = (
                multiply(self.xyaboj, 2*self.xyaboj - 1.0),
                multiply(self.yaboj, 4*self.xyaboj - 1.0),
                multiply(self.xbaoj, 4*self.xyaboj - 1.0),
                2*square(self.yaboj),
                4*multiply(self.xbaoj, self.yaboj),
                2*square(self.xbaoj)
            )
        return self._qc
    @property
    def qab(self) -> Tuple[matrix]:
        if self._qab is None:
            self._qab = (
                4*multiply(self.xybcoj, self.xycaoj),
                4*(multiply(self.xybcoj, self.ycaoj) + multiply(self.xycaoj, self.ybcoj)),
                4*(multiply(self.xacoj, self.xybcoj) + multiply(self.xcboj, self.xycaoj)),
                4*multiply(self.ybcoj, self.ycaoj),
                4*(multiply(self.xacoj, self.ybcoj) + multiply(self.xcboj, self.ycaoj)),
                4*multiply(self.xacoj, self.xcboj)
            )
        return self._qab
    @property
    def qbc(self) -> Tuple[matrix]:
        if self._qbc is None:
            self._qbc = (
               4*multiply(self.xyaboj, self.xycaoj),
               4*(multiply(self.xyaboj, self.ycaoj) + multiply(self.xycaoj, self.yaboj)),
               4*(multiply(self.xacoj, self.xyaboj) + multiply(self.xbaoj, self.xycaoj)),
               4*multiply(self.yaboj, self.ycaoj),
               4*(multiply(self.xacoj, self.yaboj) + multiply(self.xbaoj, self.ycaoj)),
               4*multiply(self.xacoj, self.xbaoj)
            )
        return self._qbc
    @property
    def qca(self) -> Tuple[matrix]:
        if self._qca is None:
            self._qca = (
                4*multiply(self.xyaboj, self.xybcoj),
                4*(multiply(self.xyaboj, self.ybcoj) + multiply(self.xybcoj, self.yaboj)),
                4*(multiply(self.xbaoj, self.xybcoj) + multiply(self.xcboj, self.xyaboj)),
                4*multiply(self.yaboj, self.ybcoj),
                4*(multiply(self.xbaoj, self.ybcoj) + multiply(self.xcboj, self.yaboj)),
                4*multiply(self.xbaoj, self.xcboj)
            )
        return self._qca
    def return_quadratic(self, valo: matrix, valx: matrix, valy: matrix,
                         valxx: matrix, valxy: matrix, valyy: matrix):
        valoxy = (valo, valx, valy, valxx, valxy, valyy)
        vala = sum([multiply(qai, vali) for qai, vali in zip(self.qa, valoxy)])
        valb = sum([multiply(qbi, vali) for qbi, vali in zip(self.qb, valoxy)])
        valc = sum([multiply(qci, vali) for qci, vali in zip(self.qc, valoxy)])
        valab = sum([multiply(qabi, vali) for qabi, vali in zip(self.qab, valoxy)])
        valbc = sum([multiply(qbci, vali) for qbci, vali in zip(self.qbc, valoxy)])
        valca = sum([multiply(qcai, vali) for qcai, vali in zip(self.qca, valoxy)])
        return vala, valb, valc, valab, valbc, valca
