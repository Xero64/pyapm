from typing import TYPE_CHECKING

from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from cupy import ElementwiseKernel, asarray

    # Create wake for compiling
    pnts = Vector.zeros((1, ))
    veca = Vector(-1.2, 1.0, 0.0)
    vecb = Vector(-0.8, -1.0, 0.0)
    dirw = Vector(1.0, -0.25, 0.0).to_unit()


    cupy_cwdp_kernel = ElementwiseKernel(
        '''float64 px, float64 py, float64 pz,
           float64 ax, float64 ay, float64 az,
           float64 bx, float64 by, float64 bz,
           float64 cx, float64 cy, float64 cz,
           float64 betx, float64 bety, float64 betz,
           float64 tol, float64 cond''',
        '''float64 phd''',
        '''

        const double fourpi = 12.566370614359172463991854;
        const double twopi = 6.283185307179586231995927;

        double rax = px - ax;
        double ray = py - ay;
        double raz = pz - az;
        double ram = sqrt(rax*rax + ray*ray + raz*raz);
        double hax = 0.0;
        double hay = 0.0;
        double haz = 0.0;
        double rar = 0.0;
        if (ram > tol) {
            rar = 1.0 / ram;
            hax = rax / ram;
            hay = ray / ram;
            haz = raz / ram;
        }

        double rbx = px - bx;
        double rby = py - by;
        double rbz = pz - bz;
        double rbm = sqrt(rbx*rbx + rby*rby + rbz*rbz);
        double hbx = 0.0;
        double hby = 0.0;
        double hbz = 0.0;
        double rbr = 0.0;
        if (rbm > tol) {
            rbr = 1.0 / rbm;
            hbx = rbx / rbm;
            hby = rby / rbm;
            hbz = rbz / rbm;
        }

        double rcx = -cx;
        double rcy = -cy;
        double rcz = -cz;
        double rcm = sqrt(rcx*rcx + rcy*rcy + rcz*rcz);
        double hcx = 0.0;
        double hcy = 0.0;
        double hcz = 0.0;
        if (rcm > tol) {
            hcx = rcx / rcm;
            hcy = rcy / rcm;
            hcz = rcz / rcm;
        }

        double axbx = hay * hbz - haz * hby;
        double axby = haz * hbx - hax * hbz;
        double axbz = hax * hby - hay * hbx;

        double bxcx = hby * hcz - hbz * hcy;
        double bxcy = hbz * hcx - hbx * hcz;
        double bxcz = hbx * hcy - hby * hcx;

        double cxax = hcy * haz - hcz * hay;
        double cxay = hcz * hax - hcx * haz;
        double cxaz = hcx * hay - hcy * hax;

        double adb = hax * hbx + hay * hby + haz * hbz;
        double bdc = hbx * hcx + hby * hcy + hbz * hcz;
        double cda = hcx * hax + hcy * hay + hcz * haz;

        double abden = 1.0 + adb;
        double abfac = 0.0;
        if (abden > tol) {
            abfac = (rar + rbr) / abden;
        }

        double bcden = 1.0 + bdc;
        double bcfac = 0.0;
        if (bcden > tol) {
            bcfac = rbr / bcden;
        }

        double caden = 1.0 + cda;
        double cafac = 0.0;
        if (caden > tol) {
            cafac = rar / caden;
        }

        double abx = bx - ax;
        double aby = by - ay;
        double abz = bz - az;

        double acx = cx;
        double acy = cy;
        double acz = cz;

        double vzx = aby * acz - abz * acy;
        double vzy = abz * acx - abx * acz;
        double vzz = abx * acy - aby * acx;

        double vzm = sqrt(vzx*vzx + vzy*vzy + vzz*vzz);

        double dzx = 0.0;
        double dzy = 0.0;
        double dzz = 0.0;
        if (vzm > tol) {
            dzx = cond * vzx / vzm;
            dzy = cond * vzy / vzm;
            dzz = cond * vzz / vzm;
        }

        if (ram < tol) {
            hax = dzx;
            hay = dzy;
            haz = dzz;
        }

        if (rbm < tol) {
            hbx = dzx;
            hby = dzy;
            hbz = dzz;
        }

        axbx = hay * hbz - haz * hby;
        axby = haz * hbx - hax * hbz;
        axbz = hax * hby - hay * hbx;

        double num = - (axbx * hcx + axby * hcy + axbz * hcz);
        double den = 1.0 + adb + bdc + cda;

        double absnum = abs(num);
        double absden = abs(den);

        if (absnum < tol) {
            num = 0.0;
        }

        if (absden < tol) {
            den = 0.0;
        }

        phd = atan2(num, den)/twopi;

        if (absnum < tol && absden < tol) {
            phd = 0.25;
        }

        if (absnum < tol) {
            phd = -cond * phd;
        }

        ''', 'cupy_cwdp')


    def cupy_cwdp(pnts: Vector, veca: Vector, vecb: Vector,
                  dirw: Vector, *, betx: float = 1.0, bety: float = 1.0,
                  betz: float = 1.0, tol: float = 1e-12,
                  cond: float = -1.0) -> 'NDArray':

        '''Cupy implementation of constant wake doublet phi.'''

        px, py, pz = asarray(pnts.x), asarray(pnts.y), asarray(pnts.z)
        ax, ay, az = asarray(veca.x), asarray(veca.y), asarray(veca.z)
        bx, by, bz = asarray(vecb.x), asarray(vecb.y), asarray(vecb.z)
        cx, cy, cz = asarray(dirw.x), asarray(dirw.y), asarray(dirw.z)

        ph = cupy_cwdp_kernel(px, py, pz, ax, ay, az, bx, by, bz,
                              cx, cy, cz, betx, bety, betz, tol, cond)
        phid = ph.get()

        return phid


    _ = cupy_cwdp(pnts, veca, vecb, dirw)


    cupy_cwdv_kernel = ElementwiseKernel(
        '''float64 px, float64 py, float64 pz,
           float64 ax, float64 ay, float64 az,
           float64 bx, float64 by, float64 bz,
           float64 cx, float64 cy, float64 cz,
           float64 betx, float64 bety, float64 betz,
           float64 tol''',
        '''float64 vxd, float64 vyd, float64 vzd''',
        '''

        const double fourpi = 12.566370614359172463991854;

        double rax = px - ax;
        double ray = py - ay;
        double raz = pz - az;
        double ram = sqrt(rax*rax + ray*ray + raz*raz);

        double hax = 0.0;
        double hay = 0.0;
        double haz = 0.0;
        double rar = 0.0;
        if (ram > tol) {
            rar = 1.0 / ram;
            hax = rax / ram;
            hay = ray / ram;
            haz = raz / ram;
        }

        double rbx = px - bx;
        double rby = py - by;
        double rbz = pz - bz;
        double rbm = sqrt(rbx*rbx + rby*rby + rbz*rbz);

        double hbx = 0.0;
        double hby = 0.0;
        double hbz = 0.0;
        double rbr = 0.0;
        if (rbm > tol) {
            rbr = 1.0 / rbm;
            hbx = rbx / rbm;
            hby = rby / rbm;
            hbz = rbz / rbm;
        }

        double rcx = -cx;
        double rcy = -cy;
        double rcz = -cz;
        double rcm = sqrt(rcx*rcx + rcy*rcy + rcz*rcz);

        double hcx = 0.0;
        double hcy = 0.0;
        double hcz = 0.0;
        if (rcm > tol) {
            hcx = rcx / rcm;
            hcy = rcy / rcm;
            hcz = rcz / rcm;
        }

        double axbx = hay * hbz - haz * hby;
        double axby = haz * hbx - hax * hbz;
        double axbz = hax * hby - hay * hbx;

        double bxcx = hby * hcz - hbz * hcy;
        double bxcy = hbz * hcx - hbx * hcz;
        double bxcz = hbx * hcy - hby * hcx;

        double cxax = hcy * haz - hcz * hay;
        double cxay = hcz * hax - hcx * haz;
        double cxaz = hcx * hay - hcy * hax;

        double abden = 1.0 + hax * hbx + hay * hby + haz * hbz;
        double abfac = 0.0;
        if (abden > tol) {
            abfac = (rar + rbr) / abden;
        }

        double bcden = 1.0 + hbx * hcx + hby * hcy + hbz * hcz;
        double bcfac = 0.0;
        if (bcden > tol) {
            bcfac = rbr / bcden;
        }

        double caden = 1.0 + hcx * hax + hcy * hay + hcz * haz;
        double cafac = 0.0;
        if (caden > tol) {
            cafac = rar / caden;
        }

        vxd = (axbx * abfac + bxcx * bcfac + cxax * cafac)/fourpi;
        vyd = (axby * abfac + bxcy * bcfac + cxay * cafac)/fourpi;
        vzd = (axbz * abfac + bxcz * bcfac + cxaz * cafac)/fourpi;

        ''', 'cupy_cwdv')


    def cupy_cwdv(pnts: Vector, veca: Vector, vecb: Vector,
                  dirw: Vector, *, betx: float = 1.0, bety: float = 1.0,
                  betz: float = 1.0, tol: float = 1e-12) -> Vector:

        px, py, pz = asarray(pnts.x), asarray(pnts.y), asarray(pnts.z)
        ax, ay, az = asarray(veca.x), asarray(veca.y), asarray(veca.z)
        bx, by, bz = asarray(vecb.x), asarray(vecb.y), asarray(vecb.z)
        cx, cy, cz = asarray(dirw.x), asarray(dirw.y), asarray(dirw.z)

        vx, vy, vz = cupy_cwdv_kernel(px, py, pz, ax, ay, az, bx, by, bz,
                                      cx, cy, cz, betx, bety, betz, tol)

        veld = Vector(vx.get(), vy.get(), vz.get())

        return veld


    _ = cupy_cwdv(pnts, veca, vecb, dirw)


    cupy_cwdf_kernel = ElementwiseKernel(
        '''float64 px, float64 py, float64 pz,
           float64 ax, float64 ay, float64 az,
           float64 bx, float64 by, float64 bz,
           float64 cx, float64 cy, float64 cz,
           float64 betx, float64 bety, float64 betz,
           float64 tol, float64 cond''',
        '''float64 phd, float64 vxd, float64 vyd, float64 vzd''',
        '''

        const double fourpi = 12.566370614359172463991854;
        const double twopi = 6.283185307179586231995927;

        double rax = px - ax;
        double ray = py - ay;
        double raz = pz - az;
        double ram = sqrt(rax*rax + ray*ray + raz*raz);

        double hax = 0.0;
        double hay = 0.0;
        double haz = 0.0;
        double rar = 0.0;
        if (ram > tol) {
            rar = 1.0 / ram;
            hax = rax / ram;
            hay = ray / ram;
            haz = raz / ram;
        }

        double rbx = px - bx;
        double rby = py - by;
        double rbz = pz - bz;
        double rbm = sqrt(rbx*rbx + rby*rby + rbz*rbz);

        double hbx = 0.0;
        double hby = 0.0;
        double hbz = 0.0;
        double rbr = 0.0;
        if (rbm > tol) {
            rbr = 1.0 / rbm;
            hbx = rbx / rbm;
            hby = rby / rbm;
            hbz = rbz / rbm;
        }

        double rcx = -cx;
        double rcy = -cy;
        double rcz = -cz;
        double rcm = sqrt(rcx*rcx + rcy*rcy + rcz*rcz);

        double hcx = 0.0;
        double hcy = 0.0;
        double hcz = 0.0;
        if (rcm > tol) {
            hcx = rcx / rcm;
            hcy = rcy / rcm;
            hcz = rcz / rcm;
        }

        double axbx = hay * hbz - haz * hby;
        double axby = haz * hbx - hax * hbz;
        double axbz = hax * hby - hay * hbx;

        double bxcx = hby * hcz - hbz * hcy;
        double bxcy = hbz * hcx - hbx * hcz;
        double bxcz = hbx * hcy - hby * hcx;

        double cxax = hcy * haz - hcz * hay;
        double cxay = hcz * hax - hcx * haz;
        double cxaz = hcx * hay - hcy * hax;

        double adb = hax * hbx + hay * hby + haz * hbz;
        double bdc = hbx * hcx + hby * hcy + hbz * hcz;
        double cda = hcx * hax + hcy * hay + hcz * haz;

        double abden = 1.0 + adb;
        double abfac = 0.0;
        if (abden > tol) {
            abfac = (rar + rbr) / abden;
        }

        double bcden = 1.0 + bdc;
        double bcfac = 0.0;
        if (bcden > tol) {
            bcfac = rbr / bcden;
        }

        double caden = 1.0 + cda;
        double cafac = 0.0;
        if (caden > tol) {
            cafac = rar / caden;
        }

        vxd = (axbx * abfac + bxcx * bcfac + cxax * cafac)/fourpi;
        vyd = (axby * abfac + bxcy * bcfac + cxay * cafac)/fourpi;
        vzd= (axbz * abfac + bxcz * bcfac + cxaz * cafac)/fourpi;

        double abx = bx - ax;
        double aby = by - ay;
        double abz = bz - az;

        double acx = cx;
        double acy = cy;
        double acz = cz;

        double vzx = aby * acz - abz * acy;
        double vzy = abz * acx - abx * acz;
        double vzz = abx * acy - aby * acx;

        double vzm = sqrt(vzx*vzx + vzy*vzy + vzz*vzz);

        double dzx = 0.0;
        double dzy = 0.0;
        double dzz = 0.0;
        if (vzm > tol) {
            dzx = cond * vzx / vzm;
            dzy = cond * vzy / vzm;
            dzz = cond * vzz / vzm;
        }

        if (ram < tol) {
            hax = dzx;
            hay = dzy;
            haz = dzz;
        }

        if (rbm < tol) {
            hbx = dzx;
            hby = dzy;
            hbz = dzz;
        }

        axbx = hay * hbz - haz * hby;
        axby = haz * hbx - hax * hbz;
        axbz = hax * hby - hay * hbx;

        double num = - (axbx * hcx + axby * hcy + axbz * hcz);
        double den = 1.0 + adb + bdc + cda;

        double absnum = abs(num);
        double absden = abs(den);

        if (absnum < tol) {
            num = 0.0;
        }

        if (absden < tol) {
            den = 0.0;
        }

        phd = atan2(num, den)/twopi;

        if (absnum < tol && absden < tol) {
            phd = 0.25;
        }

        if (absnum < tol) {
            phd = -cond * phd;
        }

        ''', 'cupy_cwdf')


    def cupy_cwdf(pnts: Vector, veca: Vector, vecb: Vector,
                  dirw: Vector, *, betx: float = 1.0, bety: float = 1.0,
                  betz: float = 1.0, tol: float = 1e-12,
                  cond: float = -1.0) -> tuple['NDArray', Vector]:

        '''Cupy implementation of constant wake doublet flow.'''

        px, py, pz = asarray(pnts.x), asarray(pnts.y), asarray(pnts.z)
        ax, ay, az = asarray(veca.x), asarray(veca.y), asarray(veca.z)
        bx, by, bz = asarray(vecb.x), asarray(vecb.y), asarray(vecb.z)
        cx, cy, cz = asarray(dirw.x), asarray(dirw.y), asarray(dirw.z)

        ph, vx, vy, vz = cupy_cwdf_kernel(px, py, pz, ax, ay, az, bx, by, bz,
                                          cx, cy, cz, betx, bety, betz, tol, cond)

        veld = Vector(vx.get(), vy.get(), vz.get())
        phid = ph.get()

        return phid, veld


    _ = cupy_cwdf(pnts, veca, vecb, dirw)


    # Create triangle for compiling
    pnts = Vector.zeros((1))
    veca = Vector(-1.0, -1.0, 0.0)
    vecb = Vector(1.0, -1.0, 0.0)
    vecc = Vector(0.0, 1.0, 0.0)


    cupy_ctdsp_kernel = ElementwiseKernel(
        '''float64 px, float64 py, float64 pz,
           float64 ax, float64 ay, float64 az,
           float64 bx, float64 by, float64 bz,
           float64 cx, float64 cy, float64 cz,
           float64 betx, float64 bety, float64 betz,
           float64 tol, float64 cond''',
        '''float64 phd, float64 phs''',
        '''

        const double fourpi = 12.566370614359172463991854;
        const double twopi = 6.283185307179586231995927;

        double rax = px - ax;
        double ray = py - ay;
        double raz = pz - az;
        double ram = sqrt(rax*rax + ray*ray + raz*raz);
        double hax = 0.0;
        double hay = 0.0;
        double haz = 0.0;
        if (ram > tol) {
            hax = rax / ram;
            hay = ray / ram;
            haz = raz / ram;
        }

        double rbx = px - bx;
        double rby = py - by;
        double rbz = pz - bz;
        double rbm = sqrt(rbx*rbx + rby*rby + rbz*rbz);
        double hbx = 0.0;
        double hby = 0.0;
        double hbz = 0.0;
        if (rbm > tol) {
            hbx = rbx / rbm;
            hby = rby / rbm;
            hbz = rbz / rbm;
        }

        double rcx = px - cx;
        double rcy = py - cy;
        double rcz = pz - cz;
        double rcm = sqrt(rcx*rcx + rcy*rcy + rcz*rcz);
        double hcx = 0.0;
        double hcy = 0.0;
        double hcz = 0.0;
        if (rcm > tol) {
            hcx = rcx / rcm;
            hcy = rcy / rcm;
            hcz = rcz / rcm;
        }

        double adb = hax * hbx + hay * hby + haz * hbz;
        double bdc = hbx * hcx + hby * hcy + hbz * hcz;
        double cda = hcx * hax + hcy * hay + hcz * haz;

        double abx = bx - ax;
        double aby = by - ay;
        double abz = bz - az;
        double abm = sqrt(abx*abx + aby*aby + abz*abz);
        double habx = 0.0;
        double haby = 0.0;
        double habz = 0.0;
        if (abm > tol) {
            habx = abx / abm;
            haby = aby / abm;
            habz = abz / abm;
        }

        double bcx = cx - bx;
        double bcy = cy - by;
        double bcz = cz - bz;
        double bcm = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
        double hbcx = 0.0;
        double hbcy = 0.0;
        double hbcz = 0.0;
        if (bcm > tol) {
            hbcx = bcx / bcm;
            hbcy = bcy / bcm;
            hbcz = bcz / bcm;
        }

        double cax = ax - cx;
        double cay = ay - cy;
        double caz = az - cz;
        double cam = sqrt(cax*cax + cay*cay + caz*caz);
        double hcax = 0.0;
        double hcay = 0.0;
        double hcaz = 0.0;
        if (cam > tol) {
            hcax = cax / cam;
            hcay = cay / cam;
            hcaz = caz / cam;
        }

        double vzx = cay * abz - caz * aby;
        double vzy = caz * abx - cax * abz;
        double vzz = cax * aby - cay * abx;
        double vzm = sqrt(vzx*vzx + vzy*vzy + vzz*vzz);
        double dzx = 0.0;
        double dzy = 0.0;
        double dzz = 0.0;
        if (vzm > tol) {
            dzx = vzx / vzm;
            dzy = vzy / vzm;
            dzz = vzz / vzm;
        }

        double dxx = habx;
        double dxy = haby;
        double dxz = habz;

        double vyx = dzy * dxz - dzz * dxy;
        double vyy = dzz * dxx - dzx * dxz;
        double vyz = dzx * dxy - dzy * dxx;
        double vym = sqrt(vyx*vyx + vyy*vyy + vyz*vyz);
        double dyx = 0.0;
        double dyy = 0.0;
        double dyz = 0.0;
        if (vym > tol) {
            dyx = vyx / vym;
            dyy = vyy / vym;
            dyz = vyz / vym;
        }

        double ya_ab = rax * habx + ray * haby + raz * habz;
        double yb_ab = rbx * habx + rby * haby + rbz * habz;
        double yb_bc = rbx * hbcx + rby * hbcy + rbz * hbcz;
        double yc_bc = rcx * hbcx + rcy * hbcy + rcz * hbcz;
        double yc_ca = rcx * hcax + rcy * hcay + rcz * hcaz;
        double ya_ca = rax * hcax + ray * hcay + raz * hcaz;

        double pab2 = ram * ram - ya_ab * ya_ab;
        if (pab2 < tol) {
            pab2 = 1.0;
        }
        double pab = sqrt(pab2);

        double pbc2 = rbm * rbm - yb_bc * yb_bc;
        if (pbc2 < tol) {
            pbc2 = 1.0;
        }
        double pbc = sqrt(pbc2);

        double pca2 = rcm * rcm - yc_ca * yc_ca;
        if (pca2 < tol) {
            pca2 = 1.0;
        }
        double pca = sqrt(pca2);

        double absya_ab = abs(ya_ab);
        double absyb_ab = abs(yb_ab);
        double absyb_bc = abs(yb_bc);
        double absyc_bc = abs(yc_bc);
        double absyc_ca = abs(yc_ca);
        double absya_ca = abs(ya_ca);

        double rpya_ab = ram + absya_ab;
        double rpyb_ab = rbm + absyb_ab;
        double rpyb_bc = rbm + absyb_bc;
        double rpyc_bc = rcm + absyc_bc;
        double rpyc_ca = rcm + absyc_ca;
        double rpya_ca = ram + absya_ca;

        double Ea_ab = 1.0;
        if (ya_ab > 0.0) {
            Ea_ab = rpya_ab / pab;
        }
        if (ya_ab < 0.0) {
            Ea_ab = pab / rpya_ab;
        }
        double Eb_ab = 1.0;
        if (yb_ab > 0.0) {
            Eb_ab = rpyb_ab / pab;
        }
        if (yb_ab < 0.0) {
            Eb_ab = pab / rpyb_ab;
        }
        double Qab = log(Ea_ab / Eb_ab);

        double Eb_bc = 1.0;
        if (yb_bc > 0.0) {
            Eb_bc = rpyb_bc / pbc;
        }
        if (yb_bc < 0.0) {
            Eb_bc = pbc / rpyb_bc;
        }
        double Ec_bc = 1.0;
        if (yc_bc > 0.0) {
            Ec_bc = rpyc_bc / pbc;
        }
        if (yc_bc < 0.0) {
            Ec_bc = pbc / rpyc_bc;
        }
        double Qbc = log(Eb_bc / Ec_bc);

        double Ec_ca = 1.0;
        if (yc_ca > 0.0) {
            Ec_ca = rpyc_ca / pca;
        }
        if (yc_ca < 0.0) {
            Ec_ca = pca / rpyc_ca;
        }
        double Ea_ca = 1.0;
        if (ya_ca > 0.0) {
            Ea_ca = rpya_ca / pca;
        }
        if (ya_ca < 0.0) {
            Ea_ca = pca / rpya_ca;
        }
        double Qca = log(Ec_ca / Ea_ca);

        double axabx = ray * habz - raz * haby;
        double axaby = raz * habx - rax * habz;
        double axabz = rax * haby - ray * habx;
        double bxbcx = rby * hbcz - rbz * hbcy;
        double bxbcy = rbz * hbcx - rbx * hbcz;
        double bxbcz = rbx * hbcy - rby * hbcx;
        double cxcax = rcy * hcaz - rcz * hcay;
        double cxcay = rcz * hcax - rcx * hcaz;
        double cxcaz = rcx * hcay - rcy * hcax;

        double Rab = axabx * dzx + axaby * dzy + axabz * dzz;
        double Rbc = bxbcx * dzx + bxbcy * dzy + bxbcz * dzz;
        double Rca = cxcax * dzx + cxcay * dzy + cxcaz * dzz;

        if (ram < tol) {
            hax = cond*dzx;
            hay = cond*dzy;
            haz = cond*dzz;
        }
        if (rbm < tol) {
            hbx = cond*dzx;
            hby = cond*dzy;
            hbz = cond*dzz;
        }
        if (rcm < tol) {
            hcx = cond*dzx;
            hcy = cond*dzy;
            hcz = cond*dzz;
        }

        double axbx = hay * hbz - haz * hby;
        double axby = haz * hbx - hax * hbz;
        double axbz = hax * hby - hay * hbx;

        double num = - (axbx * hcx + axby * hcy + axbz * hcz);
        double den = 1.0 + adb + bdc + cda;

        double absnum = abs(num);
        double absden = abs(den);

        if (absnum < tol) {
            num = 0.0;
        }

        if (absden < tol) {
            den = 0.0;
        }

        phd = atan2(num, den)/twopi;

        double adz = rax * dzx + ray * dzy + raz * dzz;

        double absadz = abs(adz);

        if (absnum < tol && absden < tol) {
            phd = 0.25;
        }

        if (absnum < tol) {
            phd = -cond * phd;
        }

        double Cab = habx * dxx + haby * dxy + habz * dxz;
        double Cbc = hbcx * dxx + hbcy * dxy + hbcz * dxz;
        double Cca = hcax * dxx + hcay * dxy + hcaz * dxz;
        double Sab = habx * dyx + haby * dyy + habz * dyz;
        double Sbc = hbcx * dyx + hbcy * dyy + hbcz * dyz;
        double Sca = hcax * dyx + hcay * dyy + hcaz * dyz;

        double vxsl = (Sab*Qab + Sbc*Qbc + Sca*Qca) / fourpi;
        double vysl = -(Cab*Qab + Cbc*Qbc + Cca*Qca) / fourpi;
        double vzsl = -phd;

        phs = -phd*adz + (Rab*Qab + Rbc*Qbc + Rca*Qca) / fourpi;

        ''', 'cupy_ctdsp')


    def cupy_ctdsp(pnts: Vector, veca: Vector, vecb: Vector,
                   vecc: Vector, *, betx: float = 1.0, bety: float = 1.0,
                   betz: float = 1.0, tol: float = 1e-12,
                   cond: float = -1.0) -> tuple['NDArray', 'NDArray']:
        """
        Cupy implementation of constant triangle doublet source phi.
        """
        px, py, pz = asarray(pnts.x), asarray(pnts.y), asarray(pnts.z)
        ax, ay, az = asarray(veca.x), asarray(veca.y), asarray(veca.z)
        bx, by, bz = asarray(vecb.x), asarray(vecb.y), asarray(vecb.z)
        cx, cy, cz = asarray(vecc.x), asarray(vecc.y), asarray(vecc.z)

        phd, phs = cupy_ctdsp_kernel(px, py, pz, ax, ay, az, bx, by, bz,
                                     cx, cy, cz, betx, bety, betz, tol, cond)

        phd = phd.get()
        phs = phs.get()

        return phd, phs


    _ = cupy_ctdsp(pnts, veca, vecb, vecc)


    cupy_ctdsv_kernel = ElementwiseKernel(
        '''float64 px, float64 py, float64 pz,
           float64 ax, float64 ay, float64 az,
           float64 bx, float64 by, float64 bz,
           float64 cx, float64 cy, float64 cz,
           float64 betx, float64 bety, float64 betz,
           float64 tol, float64 cond''',
        '''float64 vxd, float64 vyd, float64 vzd,
           float64 vxs, float64 vys, float64 vzs''',
        '''

        const double fourpi = 12.566370614359172463991854;
        const double twopi = 6.283185307179586231995927;

        double rax = px - ax;
        double ray = py - ay;
        double raz = pz - az;
        double ram = sqrt(rax*rax + ray*ray + raz*raz);
        double hax = 0.0;
        double hay = 0.0;
        double haz = 0.0;
        double rar = 0.0;
        if (ram > tol) {
            rar = 1.0 / ram;
            hax = rax / ram;
            hay = ray / ram;
            haz = raz / ram;
        }

        double rbx = px - bx;
        double rby = py - by;
        double rbz = pz - bz;
        double rbm = sqrt(rbx*rbx + rby*rby + rbz*rbz);
        double hbx = 0.0;
        double hby = 0.0;
        double hbz = 0.0;
        double rbr = 0.0;
        if (rbm > tol) {
            rbr = 1.0 / rbm;
            hbx = rbx / rbm;
            hby = rby / rbm;
            hbz = rbz / rbm;
        }

        double rcx = px - cx;
        double rcy = py - cy;
        double rcz = pz - cz;
        double rcm = sqrt(rcx*rcx + rcy*rcy + rcz*rcz);
        double hcx = 0.0;
        double hcy = 0.0;
        double hcz = 0.0;
        double rcr = 0.0;
        if (rcm > tol) {
            rcr = 1.0 / rcm;
            hcx = rcx / rcm;
            hcy = rcy / rcm;
            hcz = rcz / rcm;
        }

        double adb = hax * hbx + hay * hby + haz * hbz;
        double bdc = hbx * hcx + hby * hcy + hbz * hcz;
        double cda = hcx * hax + hcy * hay + hcz * haz;

        double abx = bx - ax;
        double aby = by - ay;
        double abz = bz - az;
        double abm = sqrt(abx*abx + aby*aby + abz*abz);
        double habx = 0.0;
        double haby = 0.0;
        double habz = 0.0;
        if (abm > tol) {
            habx = abx / abm;
            haby = aby / abm;
            habz = abz / abm;
        }

        double bcx = cx - bx;
        double bcy = cy - by;
        double bcz = cz - bz;
        double bcm = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
        double hbcx = 0.0;
        double hbcy = 0.0;
        double hbcz = 0.0;
        if (bcm > tol) {
            hbcx = bcx / bcm;
            hbcy = bcy / bcm;
            hbcz = bcz / bcm;
        }

        double cax = ax - cx;
        double cay = ay - cy;
        double caz = az - cz;
        double cam = sqrt(cax*cax + cay*cay + caz*caz);
        double hcax = 0.0;
        double hcay = 0.0;
        double hcaz = 0.0;
        if (cam > tol) {
            hcax = cax / cam;
            hcay = cay / cam;
            hcaz = caz / cam;
        }

        double vzx = cay * abz - caz * aby;
        double vzy = caz * abx - cax * abz;
        double vzz = cax * aby - cay * abx;
        double vzm = sqrt(vzx*vzx + vzy*vzy + vzz*vzz);
        double dzx = 0.0;
        double dzy = 0.0;
        double dzz = 0.0;
        if (vzm > tol) {
            dzx = vzx / vzm;
            dzy = vzy / vzm;
            dzz = vzz / vzm;
        }

        double dxx = habx;
        double dxy = haby;
        double dxz = habz;

        double vyx = dzy * dxz - dzz * dxy;
        double vyy = dzz * dxx - dzx * dxz;
        double vyz = dzx * dxy - dzy * dxx;
        double vym = sqrt(vyx*vyx + vyy*vyy + vyz*vyz);
        double dyx = 0.0;
        double dyy = 0.0;
        double dyz = 0.0;
        if (vym > tol) {
            dyx = vyx / vym;
            dyy = vyy / vym;
            dyz = vyz / vym;
        }

        double ya_ab = rax * habx + ray * haby + raz * habz;
        double yb_ab = rbx * habx + rby * haby + rbz * habz;
        double yb_bc = rbx * hbcx + rby * hbcy + rbz * hbcz;
        double yc_bc = rcx * hbcx + rcy * hbcy + rcz * hbcz;
        double yc_ca = rcx * hcax + rcy * hcay + rcz * hcaz;
        double ya_ca = rax * hcax + ray * hcay + raz * hcaz;

        double pab2 = ram * ram - ya_ab * ya_ab;
        if (pab2 < tol) {
            pab2 = 1.0;
        }
        double pab = sqrt(pab2);

        double pbc2 = rbm * rbm - yb_bc * yb_bc;
        if (pbc2 < tol) {
            pbc2 = 1.0;
        }
        double pbc = sqrt(pbc2);

        double pca2 = rcm * rcm - yc_ca * yc_ca;
        if (pca2 < tol) {
            pca2 = 1.0;
        }
        double pca = sqrt(pca2);

        double absya_ab = abs(ya_ab);
        double absyb_ab = abs(yb_ab);
        double absyb_bc = abs(yb_bc);
        double absyc_bc = abs(yc_bc);
        double absyc_ca = abs(yc_ca);
        double absya_ca = abs(ya_ca);

        double rpya_ab = ram + absya_ab;
        double rpyb_ab = rbm + absyb_ab;
        double rpyb_bc = rbm + absyb_bc;
        double rpyc_bc = rcm + absyc_bc;
        double rpyc_ca = rcm + absyc_ca;
        double rpya_ca = ram + absya_ca;

        double Ea_ab = 1.0;
        if (ya_ab > 0.0) {
            Ea_ab = rpya_ab / pab;
        }
        if (ya_ab < 0.0) {
            Ea_ab = pab / rpya_ab;
        }
        double Eb_ab = 1.0;
        if (yb_ab > 0.0) {
            Eb_ab = rpyb_ab / pab;
        }
        if (yb_ab < 0.0) {
            Eb_ab = pab / rpyb_ab;
        }
        double Qab = log(Ea_ab / Eb_ab);

        double Eb_bc = 1.0;
        if (yb_bc > 0.0) {
            Eb_bc = rpyb_bc / pbc;
        }
        if (yb_bc < 0.0) {
            Eb_bc = pbc / rpyb_bc;
        }
        double Ec_bc = 1.0;
        if (yc_bc > 0.0) {
            Ec_bc = rpyc_bc / pbc;
        }
        if (yc_bc < 0.0) {
            Ec_bc = pbc / rpyc_bc;
        }
        double Qbc = log(Eb_bc / Ec_bc);

        double Ec_ca = 1.0;
        if (yc_ca > 0.0) {
            Ec_ca = rpyc_ca / pca;
        }
        if (yc_ca < 0.0) {
            Ec_ca = pca / rpyc_ca;
        }
        double Ea_ca = 1.0;
        if (ya_ca > 0.0) {
            Ea_ca = rpya_ca / pca;
        }
        if (ya_ca < 0.0) {
            Ea_ca = pca / rpya_ca;
        }
        double Qca = log(Ec_ca / Ea_ca);

        double axbx = hay * hbz - haz * hby;
        double axby = haz * hbx - hax * hbz;
        double axbz = hax * hby - hay * hbx;

        double bxcx = hby * hcz - hbz * hcy;
        double bxcy = hbz * hcx - hbx * hcz;
        double bxcz = hbx * hcy - hby * hcx;

        double cxax = hcy * haz - hcz * hay;
        double cxay = hcz * hax - hcx * haz;
        double cxaz = hcx * hay - hcy * hax;

        double abden = 1.0 + hax * hbx + hay * hby + haz * hbz;
        double abfac = 0.0;
        if (abden > tol) {
            abfac = (rar + rbr) / abden;
        }

        double bcden = 1.0 + hbx * hcx + hby * hcy + hbz * hcz;
        double bcfac = 0.0;
        if (bcden > tol) {
            bcfac = (rbr + rcr) / bcden;
        }

        double caden = 1.0 + hcx * hax + hcy * hay + hcz * haz;
        double cafac = 0.0;
        if (caden > tol) {
            cafac = (rcr + rar) / caden;
        }

        vxd = (axbx * abfac + bxcx * bcfac + cxax * cafac)/fourpi;
        vyd = (axby * abfac + bxcy * bcfac + cxay * cafac)/fourpi;
        vzd = (axbz * abfac + bxcz * bcfac + cxaz * cafac)/fourpi;

        if (ram < tol) {
            hax = cond*dzx;
            hay = cond*dzy;
            haz = cond*dzz;
        }
        if (rbm < tol) {
            hbx = cond*dzx;
            hby = cond*dzy;
            hbz = cond*dzz;
        }
        if (rcm < tol) {
            hcx = cond*dzx;
            hcy = cond*dzy;
            hcz = cond*dzz;
        }

        axbx = hay * hbz - haz * hby;
        axby = haz * hbx - hax * hbz;
        axbz = hax * hby - hay * hbx;

        double num = - (axbx * hcx + axby * hcy + axbz * hcz);
        double den = 1.0 + adb + bdc + cda;

        double absnum = abs(num);
        double absden = abs(den);

        if (absnum < tol) {
            num = 0.0;
        }

        if (absden < tol) {
            den = 0.0;
        }

        double phd = atan2(num, den)/twopi;

        double adz = rax * dzx + ray * dzy + raz * dzz;

        double absadz = abs(adz);

        if (absnum < tol && absden < tol) {
            phd = 0.25;
        }

        if (absnum < tol) {
            phd = -cond * phd;
        }

        double Cab = habx * dxx + haby * dxy + habz * dxz;
        double Cbc = hbcx * dxx + hbcy * dxy + hbcz * dxz;
        double Cca = hcax * dxx + hcay * dxy + hcaz * dxz;
        double Sab = habx * dyx + haby * dyy + habz * dyz;
        double Sbc = hbcx * dyx + hbcy * dyy + hbcz * dyz;
        double Sca = hcax * dyx + hcay * dyy + hcaz * dyz;

        double vxsl = (Sab*Qab + Sbc*Qbc + Sca*Qca) / fourpi;
        double vysl = -(Cab*Qab + Cbc*Qbc + Cca*Qca) / fourpi;
        double vzsl = -phd;

        vxs = dxx * vxsl + dyx * vysl + dzx * vzsl;
        vys = dxy * vxsl + dyy * vysl + dzy * vzsl;
        vzs = dxz * vxsl + dyz * vysl + dzz * vzsl;

        ''', 'cupy_ctdsv')


    def cupy_ctdsv(pnts: Vector, veca: Vector, vecb: Vector,
                   vecc: Vector, *, betx: float = 1.0, bety: float = 1.0,
                   betz: float = 1.0, tol: float = 1e-12,
                   cond: float = -1.0) -> tuple[Vector, Vector]:
        """
        Cupy implementation of constant triangle doublet source velocity.
        """
        px, py, pz = asarray(pnts.x), asarray(pnts.y), asarray(pnts.z)
        ax, ay, az = asarray(veca.x), asarray(veca.y), asarray(veca.z)
        bx, by, bz = asarray(vecb.x), asarray(vecb.y), asarray(vecb.z)
        cx, cy, cz = asarray(vecc.x), asarray(vecc.y), asarray(vecc.z)

        vxd, vyd, vzd, vxs, vys, vzs = cupy_ctdsv_kernel(px, py, pz,
                                                         ax, ay, az,
                                                         bx, by, bz,
                                                         cx, cy, cz,
                                                         betx, bety, betz,
                                                         tol, cond)

        veld = Vector(vxd.get(), vyd.get(), vzd.get())
        vels = Vector(vxs.get(), vys.get(), vzs.get())

        return veld, vels


    _ = cupy_ctdsv(pnts, veca, vecb, vecc)


    cupy_ctdsf_kernel = ElementwiseKernel(
        '''float64 px, float64 py, float64 pz,
           float64 ax, float64 ay, float64 az,
           float64 bx, float64 by, float64 bz,
           float64 cx, float64 cy, float64 cz,
           float64 betx, float64 bety, float64 betz,
           float64 tol, float64 cond''',
        '''float64 phd, float64 vxd, float64 vyd, float64 vzd,
           float64 phs, float64 vxs, float64 vys, float64 vzs''',
        '''

        const double fourpi = 12.566370614359172463991854;
        const double twopi = 6.283185307179586231995927;

        double rax = px - ax;
        double ray = py - ay;
        double raz = pz - az;
        double ram = sqrt(rax*rax + ray*ray + raz*raz);
        double hax = 0.0;
        double hay = 0.0;
        double haz = 0.0;
        double rar = 0.0;
        if (ram > tol) {
            rar = 1.0 / ram;
            hax = rax / ram;
            hay = ray / ram;
            haz = raz / ram;
        }

        double rbx = px - bx;
        double rby = py - by;
        double rbz = pz - bz;
        double rbm = sqrt(rbx*rbx + rby*rby + rbz*rbz);
        double hbx = 0.0;
        double hby = 0.0;
        double hbz = 0.0;
        double rbr = 0.0;
        if (rbm > tol) {
            rbr = 1.0 / rbm;
            hbx = rbx / rbm;
            hby = rby / rbm;
            hbz = rbz / rbm;
        }

        double rcx = px - cx;
        double rcy = py - cy;
        double rcz = pz - cz;
        double rcm = sqrt(rcx*rcx + rcy*rcy + rcz*rcz);
        double hcx = 0.0;
        double hcy = 0.0;
        double hcz = 0.0;
        double rcr = 0.0;
        if (rcm > tol) {
            rcr = 1.0 / rcm;
            hcx = rcx / rcm;
            hcy = rcy / rcm;
            hcz = rcz / rcm;
        }

        double adb = hax * hbx + hay * hby + haz * hbz;
        double bdc = hbx * hcx + hby * hcy + hbz * hcz;
        double cda = hcx * hax + hcy * hay + hcz * haz;

        double abx = bx - ax;
        double aby = by - ay;
        double abz = bz - az;
        double abm = sqrt(abx*abx + aby*aby + abz*abz);
        double habx = 0.0;
        double haby = 0.0;
        double habz = 0.0;
        if (abm > tol) {
            habx = abx / abm;
            haby = aby / abm;
            habz = abz / abm;
        }

        double bcx = cx - bx;
        double bcy = cy - by;
        double bcz = cz - bz;
        double bcm = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
        double hbcx = 0.0;
        double hbcy = 0.0;
        double hbcz = 0.0;
        if (bcm > tol) {
            hbcx = bcx / bcm;
            hbcy = bcy / bcm;
            hbcz = bcz / bcm;
        }

        double cax = ax - cx;
        double cay = ay - cy;
        double caz = az - cz;
        double cam = sqrt(cax*cax + cay*cay + caz*caz);
        double hcax = 0.0;
        double hcay = 0.0;
        double hcaz = 0.0;
        if (cam > tol) {
            hcax = cax / cam;
            hcay = cay / cam;
            hcaz = caz / cam;
        }

        double vzx = cay * abz - caz * aby;
        double vzy = caz * abx - cax * abz;
        double vzz = cax * aby - cay * abx;
        double vzm = sqrt(vzx*vzx + vzy*vzy + vzz*vzz);
        double dzx = 0.0;
        double dzy = 0.0;
        double dzz = 0.0;
        if (vzm > tol) {
            dzx = vzx / vzm;
            dzy = vzy / vzm;
            dzz = vzz / vzm;
        }

        double dxx = habx;
        double dxy = haby;
        double dxz = habz;

        double vyx = dzy * dxz - dzz * dxy;
        double vyy = dzz * dxx - dzx * dxz;
        double vyz = dzx * dxy - dzy * dxx;
        double vym = sqrt(vyx*vyx + vyy*vyy + vyz*vyz);
        double dyx = 0.0;
        double dyy = 0.0;
        double dyz = 0.0;
        if (vym > tol) {
            dyx = vyx / vym;
            dyy = vyy / vym;
            dyz = vyz / vym;
        }

        double ya_ab = rax * habx + ray * haby + raz * habz;
        double yb_ab = rbx * habx + rby * haby + rbz * habz;
        double yb_bc = rbx * hbcx + rby * hbcy + rbz * hbcz;
        double yc_bc = rcx * hbcx + rcy * hbcy + rcz * hbcz;
        double yc_ca = rcx * hcax + rcy * hcay + rcz * hcaz;
        double ya_ca = rax * hcax + ray * hcay + raz * hcaz;

        double pab2 = ram * ram - ya_ab * ya_ab;
        if (pab2 < tol) {
            pab2 = 1.0;
        }
        double pab = sqrt(pab2);

        double pbc2 = rbm * rbm - yb_bc * yb_bc;
        if (pbc2 < tol) {
            pbc2 = 1.0;
        }
        double pbc = sqrt(pbc2);

        double pca2 = rcm * rcm - yc_ca * yc_ca;
        if (pca2 < tol) {
            pca2 = 1.0;
        }
        double pca = sqrt(pca2);

        double absya_ab = abs(ya_ab);
        double absyb_ab = abs(yb_ab);
        double absyb_bc = abs(yb_bc);
        double absyc_bc = abs(yc_bc);
        double absyc_ca = abs(yc_ca);
        double absya_ca = abs(ya_ca);

        double rpya_ab = ram + absya_ab;
        double rpyb_ab = rbm + absyb_ab;
        double rpyb_bc = rbm + absyb_bc;
        double rpyc_bc = rcm + absyc_bc;
        double rpyc_ca = rcm + absyc_ca;
        double rpya_ca = ram + absya_ca;

        double Ea_ab = 1.0;
        if (ya_ab > 0.0) {
            Ea_ab = rpya_ab / pab;
        }
        if (ya_ab < 0.0) {
            Ea_ab = pab / rpya_ab;
        }
        double Eb_ab = 1.0;
        if (yb_ab > 0.0) {
            Eb_ab = rpyb_ab / pab;
        }
        if (yb_ab < 0.0) {
            Eb_ab = pab / rpyb_ab;
        }
        double Qab = log(Ea_ab / Eb_ab);

        double Eb_bc = 1.0;
        if (yb_bc > 0.0) {
            Eb_bc = rpyb_bc / pbc;
        }
        if (yb_bc < 0.0) {
            Eb_bc = pbc / rpyb_bc;
        }
        double Ec_bc = 1.0;
        if (yc_bc > 0.0) {
            Ec_bc = rpyc_bc / pbc;
        }
        if (yc_bc < 0.0) {
            Ec_bc = pbc / rpyc_bc;
        }
        double Qbc = log(Eb_bc / Ec_bc);

        double Ec_ca = 1.0;
        if (yc_ca > 0.0) {
            Ec_ca = rpyc_ca / pca;
        }
        if (yc_ca < 0.0) {
            Ec_ca = pca / rpyc_ca;
        }
        double Ea_ca = 1.0;
        if (ya_ca > 0.0) {
            Ea_ca = rpya_ca / pca;
        }
        if (ya_ca < 0.0) {
            Ea_ca = pca / rpya_ca;
        }
        double Qca = log(Ec_ca / Ea_ca);

        double axabx = ray * habz - raz * haby;
        double axaby = raz * habx - rax * habz;
        double axabz = rax * haby - ray * habx;
        double bxbcx = rby * hbcz - rbz * hbcy;
        double bxbcy = rbz * hbcx - rbx * hbcz;
        double bxbcz = rbx * hbcy - rby * hbcx;
        double cxcax = rcy * hcaz - rcz * hcay;
        double cxcay = rcz * hcax - rcx * hcaz;
        double cxcaz = rcx * hcay - rcy * hcax;

        double Rab = axabx * dzx + axaby * dzy + axabz * dzz;
        double Rbc = bxbcx * dzx + bxbcy * dzy + bxbcz * dzz;
        double Rca = cxcax * dzx + cxcay * dzy + cxcaz * dzz;

        double axbx = hay * hbz - haz * hby;
        double axby = haz * hbx - hax * hbz;
        double axbz = hax * hby - hay * hbx;

        double bxcx = hby * hcz - hbz * hcy;
        double bxcy = hbz * hcx - hbx * hcz;
        double bxcz = hbx * hcy - hby * hcx;

        double cxax = hcy * haz - hcz * hay;
        double cxay = hcz * hax - hcx * haz;
        double cxaz = hcx * hay - hcy * hax;

        double abden = 1.0 + hax * hbx + hay * hby + haz * hbz;
        double abfac = 0.0;
        if (abden > tol) {
            abfac = (rar + rbr) / abden;
        }

        double bcden = 1.0 + hbx * hcx + hby * hcy + hbz * hcz;
        double bcfac = 0.0;
        if (bcden > tol) {
            bcfac = (rbr + rcr) / bcden;
        }

        double caden = 1.0 + hcx * hax + hcy * hay + hcz * haz;
        double cafac = 0.0;
        if (caden > tol) {
            cafac = (rcr + rar) / caden;
        }

        vxd = (axbx * abfac + bxcx * bcfac + cxax * cafac)/fourpi;
        vyd = (axby * abfac + bxcy * bcfac + cxay * cafac)/fourpi;
        vzd = (axbz * abfac + bxcz * bcfac + cxaz * cafac)/fourpi;

        if (ram < tol) {
            hax = cond*dzx;
            hay = cond*dzy;
            haz = cond*dzz;
        }
        if (rbm < tol) {
            hbx = cond*dzx;
            hby = cond*dzy;
            hbz = cond*dzz;
        }
        if (rcm < tol) {
            hcx = cond*dzx;
            hcy = cond*dzy;
            hcz = cond*dzz;
        }

        axbx = hay * hbz - haz * hby;
        axby = haz * hbx - hax * hbz;
        axbz = hax * hby - hay * hbx;

        double num = - (axbx * hcx + axby * hcy + axbz * hcz);
        double den = 1.0 + adb + bdc + cda;

        double absnum = abs(num);
        double absden = abs(den);

        if (absnum < tol) {
            num = 0.0;
        }

        if (absden < tol) {
            den = 0.0;
        }

        phd = atan2(num, den)/twopi;

        double adz = rax * dzx + ray * dzy + raz * dzz;

        double absadz = abs(adz);

        if (absnum < tol && absden < tol) {
            phd = 0.25;
        }

        if (absnum < tol) {
            phd = -cond * phd;
        }

        phs = -phd*adz + (Rab*Qab + Rbc*Qbc + Rca*Qca) / fourpi;

        double Cab = habx * dxx + haby * dxy + habz * dxz;
        double Cbc = hbcx * dxx + hbcy * dxy + hbcz * dxz;
        double Cca = hcax * dxx + hcay * dxy + hcaz * dxz;
        double Sab = habx * dyx + haby * dyy + habz * dyz;
        double Sbc = hbcx * dyx + hbcy * dyy + hbcz * dyz;
        double Sca = hcax * dyx + hcay * dyy + hcaz * dyz;

        double vxsl = (Sab*Qab + Sbc*Qbc + Sca*Qca) / fourpi;
        double vysl = -(Cab*Qab + Cbc*Qbc + Cca*Qca) / fourpi;
        double vzsl = -phd;

        vxs = dxx * vxsl + dyx * vysl + dzx * vzsl;
        vys = dxy * vxsl + dyy * vysl + dzy * vzsl;
        vzs = dxz * vxsl + dyz * vysl + dzz * vzsl;

        ''', 'cupy_ctdsf')


    def cupy_ctdsf(pnts: Vector, veca: Vector, vecb: Vector,
                   vecc: Vector, *, betx: float = 1.0, bety: float = 1.0,
                   betz: float = 1.0, tol: float = 1e-12,
                   cond: float = -1.0) -> tuple['NDArray', 'NDArray',
                                                Vector, Vector]:
        """
        Cupy implementation of constant triangle doublet source velocity.
        """
        px, py, pz = asarray(pnts.x), asarray(pnts.y), asarray(pnts.z)
        ax, ay, az = asarray(veca.x), asarray(veca.y), asarray(veca.z)
        bx, by, bz = asarray(vecb.x), asarray(vecb.y), asarray(vecb.z)
        cx, cy, cz = asarray(vecc.x), asarray(vecc.y), asarray(vecc.z)

        phd, vxd, vyd, vzd, phs, vxs, vys, vzs = cupy_ctdsf_kernel(px, py, pz,
                                                                   ax, ay, az,
                                                                   bx, by, bz,
                                                                   cx, cy, cz,
                                                                   betx, bety, betz,
                                                                   tol, cond)

        phd = phd.get()
        veld = Vector(vxd.get(), vyd.get(), vzd.get())
        phs = phs.get()
        vels = Vector(vxs.get(), vys.get(), vzs.get())

        return phd, phs, veld, vels


    _ = cupy_ctdsf(pnts, veca, vecb, vecc)


except ImportError:
    pass
