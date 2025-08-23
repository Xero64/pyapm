from numpy import linspace, meshgrid, multiply
from pygeom.geom3d import Vector


def point_mesh_xy(xorg: float, yorg: float, zorg: float,
                  xnum: int, ynum: int,
                  xamp: float, yamp: float) -> Vector:
    xint = 2*xamp/(xnum-1)
    yint = 2*yamp/(ynum-1)
    pnts = Vector.zeros((ynum, xnum))
    for i in range(ynum):
        for j in range(xnum):
            x = xorg-xamp+xint*j
            y = yorg-yamp+yint*i
            pnts[i, j] = Vector(x, y, zorg)
    return pnts

def point_mesh_yz(xorg: float, yorg: float, zorg: float,
                  ynum: int, znum: int,
                  yamp: float, zamp: float) -> Vector:
    yint = 2*yamp/(ynum-1)
    zint = 2*zamp/(znum-1)
    pnts = Vector.zeros((znum, ynum))
    for i in range(znum):
        for j in range(ynum):
            y = yorg-yamp+yint*j
            z = zorg-zamp+zint*i
            pnts[i, j] = Vector(xorg, y, z)
    return pnts

def point_mesh_zx(xorg: float, yorg: float, zorg: float,
                  znum: int, xnum: int,
                  zamp: float, xamp: float) -> Vector:
    zint = 2*zamp/(znum-1)
    xint = 2*xamp/(xnum-1)
    pnts = Vector.zeros((xnum, znum))
    for i in range(xnum):
        for j in range(znum):
            z = zorg-zamp+zint*j
            x = xorg-xamp+xint*i
            pnts[i, j] = Vector(x, yorg, z)
    return pnts

def bilinear_quad_mesh(pnt1: Vector, pnt2: Vector,
                       pnt3: Vector, pnt4: Vector,
                       num: int) -> Vector:
    xi = linspace(-1.0, 1.0, num)
    xi1, xi2 = meshgrid(xi, xi,)
    n1 = multiply(1-xi1, 1-xi2)/4
    n2 = multiply(1+xi1, 1-xi2)/4
    n3 = multiply(1+xi1, 1+xi2)/4
    n4 = multiply(1-xi1, 1+xi2)/4
    x = n1*pnt1.x+n2*pnt2.x+n3*pnt3.x+n4*pnt4.x
    y = n1*pnt1.y+n2*pnt2.y+n3*pnt3.y+n4*pnt4.y
    z = n1*pnt1.z+n2*pnt2.z+n3*pnt3.z+n4*pnt4.z
    return Vector(x, y, z)

def point_mesh_x(xorg: float, yorg: float, zorg: float,
                 xnum: int, xamp: float) -> Vector:
    xint = 2*xamp/(xnum-1)
    pnts = Vector.zeros((1, xnum))
    for i in range(xnum):
        x = xorg-xamp+xint*i
        pnts[0, i] = Vector(x, yorg, zorg)
    return pnts

def point_mesh_y(xorg: float, yorg: float, zorg: float,
                 ynum: int, yamp: float) -> Vector:
    yint = 2*yamp/(ynum-1)
    pnts = Vector.zeros((1, ynum))
    for i in range(ynum):
        y = yorg-yamp+yint*i
        pnts[0, i] = Vector(xorg, y, zorg)
    return pnts

def point_mesh_z(xorg: float, yorg: float, zorg: float,
                 znum: int, zamp: float) -> Vector:
    zint = 2*zamp/(znum-1)
    pnts = Vector.zeros((1, znum))
    for i in range(znum):
        z = zorg-zamp+zint*i
        pnts[0, i] = Vector(xorg, yorg, z)
    return pnts
