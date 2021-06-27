from numpy import multiply, divide, sqrt
from numpy.matlib import matrix

class TrailingConverter():
    xa: matrix = None
    xb: matrix = None
    _xba: matrix = None
    _xan: matrix = None
    _xbn: matrix = None
    def __init__(self, xa: matrix, xb: matrix):
        self.xa = xa
        self.xb = xb
    @property
    def xba(self):
        if self._xba is None:
            self._xba = self.xb-self.xa
        return self._xba
    @property
    def xan(self):
        if self._xan is None:
            self._xan = divide(self.xa, self.xba)
        return self._xan
    @property
    def xbn(self):
        if self._xbn is None:
            self._xbn = divide(self.xb, self.xba)
        return self._xbn
    def return_linear(self, valo: matrix, valx: matrix):
        vala = divide(multiply(valo, self.xb)-valx, self.xba)
        valb = divide(valx-multiply(valo, self.xa), self.xba)
        return vala, valb
    def return_quadratic(self, valo: matrix, valx: matrix, valxx: matrix):
        phia_o = multiply(self.xbn, self.xan + self.xbn)
        phia_x = -(self.xan + 3*self.xbn)
        phia_xx = 2
        phib_o = multiply(self.xan, self.xan + self.xbn)
        phib_x = -(3*self.xan + self.xbn)
        phib_xx = 2
        phiab_o = -4*multiply(self.xa, self.xb)
        phiab_x = 4*(self.xan + self.xbn)
        phiab_xx = -4
        vala = multiply(phia_o, valo) + multiply(phia_x, valx) + multiply(phia_xx, valxx)
        valb = multiply(phib_o, valo) + multiply(phib_x, valx) + multiply(phib_xx, valxx)
        valab = multiply(phiab_o, valo) + multiply(phiab_x, valx) + multiply(phiab_xx, valxx)
        return vala, valb, valab

class BoundConverter():
    xa: matrix = None
    ya: matrix = None
    xb: matrix = None
    yb: matrix = None
    dx: float = None
    dy: float = None
    _xba: matrix = None
    _yab: matrix = None
    _jac: matrix = None
    def __init__(self, xa: matrix, ya: matrix, xb: matrix, yb: matrix,
                 dx: float, dy: float):
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        l = sqrt(dx**2+dy**2)
        self.dx = dx/l
        self.dy = dy/l
    @property
    def xba(self):
        if self._xba is None:
            self._xba = self.xb-self.xa
        return self._xba
    @property
    def yab(self):
        if self._yab is None:
            self._yab = self.ya-self.yb
        return self._yab
    @property
    def jac(self):
        if self._jac is None:
            self._jac = self.dx*self.yab - self.dy*self.xba
        return self._jac
    def return_linear(self, valo: matrix, valx: matrix, valy: matrix):
        vala = (multiply(self.dy*self.xb-self.dx*self.yb, valo)-self.dy*valx+self.dx*valy)/self.jac
        valb = (multiply(self.dx*self.ya-self.dy*self.xa, valo)+self.dy*valx-self.dx*valy)/self.jac
        return vala, valb
    def return_quadratic(self, valo: matrix, valy: matrix, valyy: matrix):
        pass
        # return vala, valb, valab
