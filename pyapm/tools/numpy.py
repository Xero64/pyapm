from typing import TYPE_CHECKING

from numpy import (absolute, arctan2, divide, full, log, logical_and, ones, pi,
                   reciprocal, shape, sign, sqrt, square, where, zeros)
from pygeom.geom3d import Transform, Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


FOURPI = 4.0*pi
TWOPI = 2.0*pi


def numpy_cwdp(pnts: Vector, veca: Vector, vecb: Vector,
               dirw: Vector, *, tol: float = 1e-12, betx: float = 1.0,
               bety: float = 1.0, betz: float = 1.0,
               cond: float = 0.0) -> 'NDArray':

    vecab = vecb - veca
    vecac = dirw
    vecz = vecab.cross(vecac)
    dirz = vecz.to_unit()

    a = pnts - veca
    b = pnts - vecb

    a.x = a.x/betx
    a.y = a.y/bety
    a.z = a.z/betz

    b.x = b.x/betx
    b.y = b.y/bety
    b.z = b.z/betz

    ah, am = a.to_unit(True)
    bh, bm = b.to_unit(True)
    ch = dirw.to_unit().__neg__()

    amnotzero = am > tol
    bmnotzero = bm > tol

    ar = zeros(shape(am))
    reciprocal(am, where=amnotzero, out=ar)

    br = zeros(shape(bm))
    reciprocal(bm, where=bmnotzero, out=br)

    adb = ah.dot(bh)
    bdc = bh.dot(ch)
    cda = ch.dot(ah)

    ah.x = where(amnotzero, ah.x, cond*dirz.x)
    ah.y = where(amnotzero, ah.y, cond*dirz.y)
    ah.z = where(amnotzero, ah.z, cond*dirz.z)
    bh.x = where(bmnotzero, bh.x, cond*dirz.x)
    bh.y = where(bmnotzero, bh.y, cond*dirz.y)
    bh.z = where(bmnotzero, bh.z, cond*dirz.z)

    num = bh.cross(ah).dot(ch)
    den = adb + bdc + cda + 1.0

    numiszero = absolute(num) < tol
    deniszero = absolute(den) < tol

    num = where(numiszero, 0.0, num)
    den = where(deniszero, 0.0, den)

    edgecheck = logical_and(numiszero, deniszero)

    phid = arctan2(num, den)/TWOPI
    phid = where(edgecheck, 0.25, phid)
    phid = where(numiszero, -cond*phid, phid)

    return phid


def numpy_cwdv(pnts: Vector, veca: Vector, vecb: Vector,
               dirw: Vector, *, betx: float = 1.0, bety: float = 1.0,
               betz: float = 1.0, tol: float = 1e-12) -> Vector:

    dirw = dirw.to_unit()

    a = pnts - veca
    b = pnts - vecb

    a.x = a.x/betx
    a.y = a.y/bety
    a.z = a.z/betz

    b.x = b.x/betx
    b.y = b.y/bety
    b.z = b.z/betz

    ah, am = a.to_unit(True)
    bh, bm = b.to_unit(True)
    ch = dirw.to_unit().__neg__()

    amnotzero = am > tol
    bmnotzero = bm > tol

    ar = zeros(am.shape)
    reciprocal(am, where=amnotzero, out=ar)

    br = zeros(bm.shape)
    reciprocal(bm, where=bmnotzero, out=br)

    axb = ah.cross(bh)
    bxc = bh.cross(ch)
    cxa = ch.cross(ah)

    adb = ah.dot(bh)
    bdc = bh.dot(ch)
    cda = ch.dot(ah)

    abd = adb + 1.0
    bcd = bdc + 1.0
    cad = cda + 1.0

    facab = zeros(axb.shape)
    divide(ar + br, abd, where=abd > tol, out=facab)
    velab = axb*facab

    facbc = zeros(bm.shape)
    divide(br, bcd, where=bcd > tol, out=facbc)
    velbc = bxc*facbc

    facca = zeros(am.shape)
    divide(ar, cad, where=cad > tol, out=facca)
    velca = cxa*facca

    veld = (velab + velbc + velca)/FOURPI

    return veld


def numpy_cwdf(pnts: Vector, veca: Vector, vecb: Vector,
               dirw: Vector, *, betx: float = 1.0, bety: float = 1.0,
               betz: float = 1.0, tol: float = 1e-12,
               cond: float = 0.0) -> tuple['NDArray', Vector]:

    vecab = vecb - veca
    vecac = dirw
    vecz = vecab.cross(vecac)
    dirz = vecz.to_unit()

    a = pnts - veca
    b = pnts - vecb

    a.x = a.x/betx
    a.y = a.y/bety
    a.z = a.z/betz

    b.x = b.x/betx
    b.y = b.y/bety
    b.z = b.z/betz

    ah, am = a.to_unit(True)
    bh, bm = b.to_unit(True)
    ch = dirw.to_unit().__neg__()

    amnotzero = am > tol
    bmnotzero = bm > tol

    ar = zeros(shape(am))
    reciprocal(am, where=amnotzero, out=ar)

    br = zeros(shape(bm))
    reciprocal(bm, where=bmnotzero, out=br)

    axb = ah.cross(bh)
    bxc = bh.cross(ch)
    cxa = ch.cross(ah)

    adb = ah.dot(bh)
    bdc = bh.dot(ch)
    cda = ch.dot(ah)

    abd = adb + 1.0
    bcd = bdc + 1.0
    cad = cda + 1.0

    facab = zeros(axb.shape)
    divide(ar + br, abd, where=abd > tol, out=facab)
    velab = axb*facab

    facbc = zeros(bm.shape)
    divide(br, bcd, where=bcd > tol, out=facbc)
    velbc = bxc*facbc

    facca = zeros(am.shape)
    divide(ar, cad, where=cad > tol, out=facca)
    velca = cxa*facca

    veld = (velab + velbc + velca)/FOURPI

    ah.x = where(amnotzero, ah.x, cond*dirz.x)
    ah.y = where(amnotzero, ah.y, cond*dirz.y)
    ah.z = where(amnotzero, ah.z, cond*dirz.z)
    bh.x = where(bmnotzero, bh.x, cond*dirz.x)
    bh.y = where(bmnotzero, bh.y, cond*dirz.y)
    bh.z = where(bmnotzero, bh.z, cond*dirz.z)

    num = bh.cross(ah).dot(ch)
    den = adb + bdc + cda + 1.0

    numiszero = absolute(num) < tol
    deniszero = absolute(den) < tol

    num = where(numiszero, 0.0, num)
    den = where(deniszero, 0.0, den)

    edgecheck = logical_and(numiszero, deniszero)

    phid = arctan2(num, den)/TWOPI
    phid = where(edgecheck, 0.25, phid)
    phid = where(numiszero, -cond*phid, phid)

    return phid, veld


def calculate_Q(ra: 'NDArray', ya: 'NDArray',
                rb: 'NDArray', yb: 'NDArray',
                *, tol: float = 1e-12) -> 'NDArray':
    # Qa = log(Ea)
    pa2 = square(ra) - square(ya)
    pa2 = where(pa2 < tol, 1.0, pa2)
    pa: 'NDArray' = sqrt(pa2)
    absya = absolute(ya)
    rpya = ra + absya
    Ea = ones(pa.shape)
    divide(rpya, pa, where=ya > 0.0, out=Ea)
    divide(pa, rpya, where=ya < 0.0, out=Ea)
    # Qb = log(Eb)
    pb2 = square(rb) - square(yb)
    pb2 = where(pb2 < tol, 1.0, pb2)
    pb: 'NDArray' = sqrt(pb2)
    absyb = absolute(yb)
    rpyb = rb + absyb
    Eb = ones(pb.shape)
    divide(rpyb, pb, where=yb > 0.0, out=Eb)
    divide(pb, rpyb, where=yb < 0.0, out=Eb)
    # Q = Qb - Qa = log(Eb) - log(Ea) = log(Eb/Ea)
    Qab = log(divide(Ea, Eb))
    return Qab


def numpy_ctdsp(pnts: Vector, veca: Vector, vecb: Vector,
                vecc: Vector, *, betx: float = 1.0, bety: float = 1.0,
                betz: float = 1.0, tol: float = 1e-12,
                cond: float = -1.0) -> tuple['NDArray', 'NDArray']:

    a = pnts - veca
    b = pnts - vecb
    c = pnts - vecc

    a.x = a.x/betx
    a.y = a.y/bety
    a.z = a.z/betz

    b.x = b.x/betx
    b.y = b.y/bety
    b.z = b.z/betz

    c.x = c.x/betx
    c.y = c.y/bety
    c.z = c.z/betz

    ah, am = a.to_unit(True)
    bh, bm = b.to_unit(True)
    ch, cm = c.to_unit(True)

    adb = ah.dot(bh)
    bdc = bh.dot(ch)
    cda = ch.dot(ah)

    vecab = vecb - veca
    vecbc = vecc - vecb
    vecca = veca - vecc

    dirab = vecab.to_unit()
    dirbc = vecbc.to_unit()
    dirca = vecca.to_unit()

    tfm = Transform(dirab, dirbc)
    dirz = tfm.dirz

    Qab = calculate_Q(am, a.dot(dirab), bm, b.dot(dirab), tol=tol)
    Qbc = calculate_Q(bm, b.dot(dirbc), cm, c.dot(dirbc), tol=tol)
    Qca = calculate_Q(cm, c.dot(dirca), am, a.dot(dirca), tol=tol)

    Rab = a.cross(dirab).dot(tfm.dirz)
    Rbc = b.cross(dirbc).dot(tfm.dirz)
    Rca = c.cross(dirca).dot(tfm.dirz)

    adz = a.dot(tfm.dirz)

    absadz = absolute(adz)

    sgnz = full(adz.shape, cond, dtype=adz.dtype)
    sign(adz, where=absadz > tol, out=sgnz)

    amnotzero = am > tol
    bmnotzero = bm > tol
    cmnotzero = cm > tol

    ah.x = where(amnotzero, ah.x, cond*dirz.x)
    ah.y = where(amnotzero, ah.y, cond*dirz.y)
    ah.z = where(amnotzero, ah.z, cond*dirz.z)
    bh.x = where(bmnotzero, bh.x, cond*dirz.x)
    bh.y = where(bmnotzero, bh.y, cond*dirz.y)
    bh.z = where(bmnotzero, bh.z, cond*dirz.z)
    ch.x = where(cmnotzero, ch.x, cond*dirz.x)
    ch.y = where(cmnotzero, ch.y, cond*dirz.y)
    ch.z = where(cmnotzero, ch.z, cond*dirz.z)

    num = bh.cross(ah).dot(ch)
    den = adb + bdc + cda + 1.0

    numiszero = absolute(num) < tol
    deniszero = absolute(den) < tol

    num = where(numiszero, 0.0, num)
    den = where(deniszero, 0.0, den)

    edgecheck = logical_and(numiszero, deniszero)

    phid = arctan2(num, den)/TWOPI
    phid = where(edgecheck, 0.25, phid)
    phid = where(numiszero, -cond*phid, phid)

    phis = -phid*adz + (Rab*Qab + Rbc*Qbc + Rca*Qca)/FOURPI

    return phid, phis


def numpy_ctdsv(pnts: Vector, veca: Vector, vecb: Vector,
                vecc: Vector, *, betx: float = 1.0, bety: float = 1.0,
                betz: float = 1.0, tol: float = 1e-12,
                cond: float = 0.0) -> tuple[Vector, Vector]:

    a = pnts - veca
    b = pnts - vecb
    c = pnts - vecc

    a.x = a.x/betx
    a.y = a.y/bety
    a.z = a.z/betz

    b.x = b.x/betx
    b.y = b.y/bety
    b.z = b.z/betz

    c.x = c.x/betx
    c.y = c.y/bety
    c.z = c.z/betz

    ah, am = a.to_unit(True)
    bh, bm = b.to_unit(True)
    ch, cm = c.to_unit(True)

    amnotzero = am > tol
    bmnotzero = bm > tol
    cmnotzero = cm > tol

    ar = zeros(shape(am))
    ar = reciprocal(am, where=amnotzero, out=ar)

    br = zeros(shape(bm))
    br = reciprocal(bm, where=bmnotzero, out=br)

    cr = zeros(shape(cm))
    cr = reciprocal(cm, where=cmnotzero, out=cr)

    axb = ah.cross(bh)
    bxc = bh.cross(ch)
    cxa = ch.cross(ah)

    adb = ah.dot(bh)
    bdc = bh.dot(ch)
    cda = ch.dot(ah)

    abd = adb + 1.0
    bcd = bdc + 1.0
    cad = cda + 1.0

    facab = zeros(axb.shape)
    divide(ar + br, abd, where=abd > tol, out=facab)
    velab = axb*facab

    facbc = zeros(bm.shape)
    divide(br + cr, bcd, where=bcd > tol, out=facbc)
    velbc = bxc*facbc

    facca = zeros(am.shape)
    divide(cr + ar, cad, where=cad > tol, out=facca)
    velca = cxa*facca

    veld = (velab + velbc + velca)/FOURPI

    vecab = vecb - veca
    vecbc = vecc - vecb
    vecca = veca - vecc

    dirab = vecab.to_unit()
    dirbc = vecbc.to_unit()
    dirca = vecca.to_unit()

    tfm = Transform(dirab, dirbc)
    dirz = tfm.dirz

    Qab = calculate_Q(am, a.dot(dirab), bm, b.dot(dirab), tol=tol)
    Qbc = calculate_Q(bm, b.dot(dirbc), cm, c.dot(dirbc), tol=tol)
    Qca = calculate_Q(cm, c.dot(dirca), am, a.dot(dirca), tol=tol)

    adz = a.dot(tfm.dirz)

    absadz = absolute(adz)

    sgnz = full(adz.shape, cond, dtype=adz.dtype)
    sign(adz, where=absadz > tol, out=sgnz)

    amnotzero = am > tol
    bmnotzero = bm > tol
    cmnotzero = cm > tol

    ah.x = where(amnotzero, ah.x, cond*dirz.x)
    ah.y = where(amnotzero, ah.y, cond*dirz.y)
    ah.z = where(amnotzero, ah.z, cond*dirz.z)
    bh.x = where(bmnotzero, bh.x, cond*dirz.x)
    bh.y = where(bmnotzero, bh.y, cond*dirz.y)
    bh.z = where(bmnotzero, bh.z, cond*dirz.z)
    ch.x = where(cmnotzero, ch.x, cond*dirz.x)
    ch.y = where(cmnotzero, ch.y, cond*dirz.y)
    ch.z = where(cmnotzero, ch.z, cond*dirz.z)

    num = bh.cross(ah).dot(ch)
    den = adb + bdc + cda + 1.0

    numiszero = absolute(num) < tol
    deniszero = absolute(den) < tol

    num = where(numiszero, 0.0, num)
    den = where(deniszero, 0.0, den)

    edgecheck = logical_and(numiszero, deniszero)

    phid = arctan2(num, den)/TWOPI
    phid = where(edgecheck, 0.25, phid)
    phid = where(numiszero, -cond*phid, phid)

    Cab = dirab.dot(tfm.dirx)
    Cbc = dirbc.dot(tfm.dirx)
    Cca = dirca.dot(tfm.dirx)
    Sab = dirab.dot(tfm.diry)
    Sbc = dirbc.dot(tfm.diry)
    Sca = dirca.dot(tfm.diry)

    velxl = (Sab*Qab + Sbc*Qbc + Sca*Qca)/FOURPI
    velyl = -(Cab*Qab + Cbc*Qbc + Cca*Qca)/FOURPI
    velzl = -phid

    vell = Vector(velxl, velyl, velzl)

    vels = tfm.vector_to_global(vell)

    return veld, vels


def numpy_ctdsf(pnts: Vector, veca: Vector, vecb: Vector,
                vecc: Vector, *, betx: float = 1.0, bety: float = 1.0,
                betz: float = 1.0, tol: float = 1e-12,
                cond: float = 0.0) -> tuple['NDArray', 'NDArray',
                                            Vector, Vector]:

    a = pnts - veca
    b = pnts - vecb
    c = pnts - vecc

    a.x = a.x/betx
    a.y = a.y/bety
    a.z = a.z/betz

    b.x = b.x/betx
    b.y = b.y/bety
    b.z = b.z/betz

    c.x = c.x/betx
    c.y = c.y/bety
    c.z = c.z/betz

    ah, am = a.to_unit(True)
    bh, bm = b.to_unit(True)
    ch, cm = c.to_unit(True)

    amnotzero = am > tol
    bmnotzero = bm > tol
    cmnotzero = cm > tol

    ar = zeros(shape(am))
    ar = reciprocal(am, where=amnotzero, out=ar)

    br = zeros(shape(bm))
    br = reciprocal(bm, where=bmnotzero, out=br)

    cr = zeros(shape(cm))
    cr = reciprocal(cm, where=cmnotzero, out=cr)

    axb = ah.cross(bh)
    bxc = bh.cross(ch)
    cxa = ch.cross(ah)

    adb = ah.dot(bh)
    bdc = bh.dot(ch)
    cda = ch.dot(ah)

    abd = adb + 1.0
    bcd = bdc + 1.0
    cad = cda + 1.0

    facab = zeros(axb.shape)
    divide(ar + br, abd, where=abd > tol, out=facab)
    velab = axb*facab

    facbc = zeros(bm.shape)
    divide(br + cr, bcd, where=bcd > tol, out=facbc)
    velbc = bxc*facbc

    facca = zeros(am.shape)
    divide(cr + ar, cad, where=cad > tol, out=facca)
    velca = cxa*facca

    veld = (velab + velbc + velca)/FOURPI

    vecab = vecb - veca
    vecbc = vecc - vecb
    vecca = veca - vecc

    dirab = vecab.to_unit()
    dirbc = vecbc.to_unit()
    dirca = vecca.to_unit()

    tfm = Transform(dirab, dirbc)
    dirz = tfm.dirz

    Qab = calculate_Q(am, a.dot(dirab), bm, b.dot(dirab), tol=tol)
    Qbc = calculate_Q(bm, b.dot(dirbc), cm, c.dot(dirbc), tol=tol)
    Qca = calculate_Q(cm, c.dot(dirca), am, a.dot(dirca), tol=tol)

    Rab = a.cross(dirab).dot(tfm.dirz)
    Rbc = b.cross(dirbc).dot(tfm.dirz)
    Rca = c.cross(dirca).dot(tfm.dirz)

    adz = a.dot(tfm.dirz)

    absadz = absolute(adz)

    sgnz = full(adz.shape, cond, dtype=adz.dtype)
    sign(adz, where=absadz > tol, out=sgnz)

    amnotzero = am > tol
    bmnotzero = bm > tol
    cmnotzero = cm > tol

    ah.x = where(amnotzero, ah.x, cond*dirz.x)
    ah.y = where(amnotzero, ah.y, cond*dirz.y)
    ah.z = where(amnotzero, ah.z, cond*dirz.z)
    bh.x = where(bmnotzero, bh.x, cond*dirz.x)
    bh.y = where(bmnotzero, bh.y, cond*dirz.y)
    bh.z = where(bmnotzero, bh.z, cond*dirz.z)
    ch.x = where(cmnotzero, ch.x, cond*dirz.x)
    ch.y = where(cmnotzero, ch.y, cond*dirz.y)
    ch.z = where(cmnotzero, ch.z, cond*dirz.z)

    num = bh.cross(ah).dot(ch)
    den = adb + bdc + cda + 1.0

    numiszero = absolute(num) < tol
    deniszero = absolute(den) < tol

    num = where(numiszero, 0.0, num)
    den = where(deniszero, 0.0, den)

    edgecheck = logical_and(numiszero, deniszero)

    phid = arctan2(num, den)/TWOPI
    phid = where(edgecheck, 0.25, phid)
    phid = where(numiszero, -cond*phid, phid)

    phis = -phid*adz + (Rab*Qab + Rbc*Qbc + Rca*Qca)/FOURPI

    Cab = dirab.dot(tfm.dirx)
    Cbc = dirbc.dot(tfm.dirx)
    Cca = dirca.dot(tfm.dirx)
    Sab = dirab.dot(tfm.diry)
    Sbc = dirbc.dot(tfm.diry)
    Sca = dirca.dot(tfm.diry)

    velxl = (Sab*Qab + Sbc*Qbc + Sca*Qca)/FOURPI
    velyl = -(Cab*Qab + Cbc*Qbc + Cca*Qca)/FOURPI
    velzl = -phid

    vell = Vector(velxl, velyl, velzl)

    vels = tfm.vector_to_global(vell)

    return phid, phis, veld, vels
