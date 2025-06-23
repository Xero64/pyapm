from typing import TYPE_CHECKING, Any

from matplotlib.pyplot import figure
from numpy import cos, pi, radians, sin, square, zeros
from py2md.classes import MDHeading, MDReport, MDTable
from pygeom.geom2d import Vector2D
from pygeom.geom3d import Coordinate, Vector
from ..tools.mass import Mass

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .panelsystem import PanelSystem as System


class PanelResult():
    name: str = None
    sys: 'System' = None
    rho: float = None
    mach: float = None
    speed: float = None
    alpha: float = None
    beta: float = None
    pbo2v: float = None
    qco2v: float = None
    rbo2v: float = None
    ctrls: dict[str, float] = None
    rcg: Vector = None
    mass: 'Mass' = None
    _acs: Coordinate = None
    _scs: Coordinate = None
    _dacsa: dict[str, Vector] = None
    _dacsb: dict[str, Vector] = None
    _dscsa: dict[str, Vector] = None
    _dscsb: dict[str, Vector] = None
    _vfs: Vector = None
    _dvfsa: Vector = None
    _dvfsb: Vector = None
    _pqr: Vector = None
    _ofs: Vector = None
    _qfs: float = None
    _arm: Vector = None
    _unsig: 'NDArray' = None
    _unmu: 'NDArray' = None
    _unphi: 'NDArray' = None
    _unnvg: 'NDArray' = None
    _sig: 'NDArray' = None
    _mu: 'NDArray' = None
    _phi: 'NDArray' = None
    _nvg: 'NDArray' = None
    _mug: 'NDArray' = None
    _sigg: 'NDArray' = None
    _qloc: Vector2D = None
    _qs: 'NDArray' = None
    _cp: 'NDArray' = None
    _nfres: 'NearFieldResult' = None
    _strpres: 'StripResult' = None
    _ffres: 'FarFieldResult' = None
    _stres: 'StabilityResult' = None
    _ctresp: 'NearFieldResult' = None
    _ctresn: 'NearFieldResult' = None
    _vfsg: Vector = None
    _vfsl: Vector = None

    def __init__(self, name: str, sys: 'System') -> None:
        self.name = name
        self.sys = sys
        self.initialise()

    def initialise(self) -> None:
        self.rho = 1.0
        self.mach = 0.0
        self.speed = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        self.pbo2v = 0.0
        self.qco2v = 0.0
        self.rbo2v = 0.0
        self.ctrls = {}
        for control in self.sys.ctrls:
            self.ctrls[control] = 0.0
        self.rcg = self.sys.rref

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def set_density(self, rho: float) -> None:
        self.rho = rho
        self.reset()

    def set_state(self, mach: float | None = None, speed: float | None = None,
                  alpha: float | None = None, beta: float | None = None,
                  pbo2v: float | None = None, qco2v: float | None = None,
                  rbo2v: float | None = None) -> None:
        if mach is not None:
            self.mach = mach
        if speed is not None:
            self.speed = speed
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if pbo2v is not None:
            self.pbo2v = pbo2v
        if qco2v is not None:
            self.qco2v = qco2v
        if rbo2v is not None:
            self.rbo2v = rbo2v
        self.reset()

    def set_controls(self, **kwargs: dict[str, float]) -> None:
        for control in kwargs:
            self.ctrls[control] = kwargs[control]
        self.reset()

    def set_cg(self, rcg: Vector) -> None:
        self.rcg = rcg
        self.reset()

    def calc_coordinate_systems(self) -> None:
        pnt = self.rcg
        cosal, sinal = trig_angle(self.alpha)
        cosbt, sinbt = trig_angle(self.beta)
        self._vfs = Vector(cosal*cosbt, -sinbt, sinal*cosbt)*self.speed
        self._dvfsa = Vector(-sinal*cosbt, 0.0, cosal*cosbt)*self.speed
        self._dvfsb = Vector(-cosal*sinbt, -cosbt, -sinal*sinbt)*self.speed
        # Aerodynamic Coordinate System
        # acs_dirx = Vector(cosal*cosbt, -sinbt, sinal*cosbt)
        # acs_diry = Vector(cosal*sinbt, cosbt, sinal*sinbt)
        acs_dirx = Vector(cosal, 0.0, sinal)
        acs_diry = Vector(0.0, 1.0, 0.0)
        self._acs = Coordinate(pnt, acs_dirx, acs_diry)
        # Stability Coordinate System
        scs_dirx = Vector(-cosal, 0.0, -sinal)
        scs_diry = Vector(0.0, 1.0, 0.0)
        self._scs = Coordinate(pnt, scs_dirx, scs_diry)
        # Derivative of Aerodynamic Coordinate System wrt alpha
        # self._dacsa = {'x': Vector(-sinal*cosbt, 0.0, cosal*cosbt),
        #                'y': Vector(-sinal*sinbt, 0.0, cosal*sinbt),
        #                'z': Vector(-cosal, 0.0, -sinal)}
        self._dacsa = {'x': Vector(-sinal, 0.0, cosal),
                       'y': Vector(0.0, 0.0, 0.0),
                       'z': Vector(-cosal, 0.0, -sinal)}
        # Derivative of Aerodynamic Coordinate System wrt beta
        # self._dacsb = {'x': Vector(-cosal*sinbt, -cosbt, -sinal*sinbt),
        #                'y': Vector(cosal*cosbt, -sinbt, sinal*cosbt),
        #                'z': Vector(0.0, 0.0, 0.0)}
        self._dacsb = {'x': Vector.zeros(),
                       'y': Vector.zeros(),
                       'z': Vector.zeros()}
        # Derivative of Stability Coordinate System wrt alpha
        self._dscsa = {'x': Vector(sinal, 0.0, -cosal),
                       'y': Vector(0.0, 0.0, 0.0),
                       'z': Vector(cosal, 0.0, sinal)}
        # Derivative of Stability Coordinate System wrt beta
        self._dscsb = {'x': Vector.zeros(),
                       'y': Vector.zeros(),
                       'z': Vector.zeros()}

    @property
    def acs(self) -> Coordinate:
        if self._acs is None:
            self.calc_coordinate_systems()
        return self._acs

    @property
    def scs(self) -> Coordinate:
        if self._scs is None:
            self.calc_coordinate_systems()
        return self._scs

    @property
    def dacsa(self) -> dict[str, Vector]:
        if self._dacsa is None:
            self.calc_coordinate_systems()
        return self._dacsa

    @property
    def dacsb(self) -> dict[str, Vector]:
        if self._dacsb is None:
            self.calc_coordinate_systems()
        return self._dacsb

    @property
    def dscsa(self) -> dict[str, Vector]:
        if self._dscsa is None:
            self.calc_coordinate_systems()
        return self._dscsa

    @property
    def dscsb(self) -> dict[str, Vector]:
        if self._dscsb is None:
            self.calc_coordinate_systems()
        return self._dscsb

    @property
    def vfs(self) -> Vector:
        if self._vfs is None:
            self.calc_coordinate_systems()
        return self._vfs

    @property
    def dvfsa(self) -> Vector:
        if self._dvfsa is None:
            self.calc_coordinate_systems()
        return self._dvfsa

    @property
    def dvfsb(self) -> Vector:
        if self._dvfsb is None:
            self.calc_coordinate_systems()
        return self._dvfsb

    @property
    def pqr(self) -> Vector:
        if self._pqr is None:
            p = self.pbo2v*2*self.speed/self.sys.bref
            q = self.qco2v*2*self.speed/self.sys.cref
            r = self.rbo2v*2*self.speed/self.sys.bref
            self._pqr = Vector(p, q, r)
        return self._pqr

    @property
    def ofs(self) -> Vector:
        if self._ofs is None:
            self._ofs = self.scs.vector_to_global(self.pqr)
        return self._ofs

    @property
    def qfs(self) -> float:
        if self._qfs is None:
            self._qfs = self.rho*self.speed**2/2
        return self._qfs

    @property
    def arm(self) -> Vector:
        if self._arm is None:
            self._arm = self.sys.pnts - self.rcg
        return self._arm

    @property
    def vfsg(self) -> Vector:
        if self._vfsg is None:
            self._vfsg = self.vfs - self.ofs.cross(self.sys.rrel)
        return self._vfsg

    @property
    def vfsl(self) -> Vector:
        if self._vfsl is None:
            self._vfsl = Vector.zeros(self.sys.numpnl)
            for pnl in self.sys.pnls.values():
                self._vfsl[pnl.ind] = pnl.crd.vector_to_local(self.vfsg[pnl.ind])
        return self._vfsl

    @property
    def qfs(self) -> float:
        if self._qfs is None:
            self._qfs = self.rho*self.speed**2/2
        return self._qfs

    @property
    def unsig(self) -> Vector:
        if self._unsig is None:
            self._unsig = self.sys.unsig(self.mach)
        return self._unsig

    @property
    def unmu(self) -> Vector:
        if self._unmu is None:
            self._unmu = self.sys.unmu(self.mach)
        return self._unmu

    @property
    def unphi(self) -> Vector:
        if self._unphi is None:
            self._unphi = self.sys.unphi(self.mach)
        return self._unphi

    @property
    def unnvg(self) -> Vector:
        if self._unnvg is None:
            self._unnvg = self.sys.unnvg(self.mach)
        return self._unnvg

    @property
    def sig(self) -> 'NDArray':
        if self._sig is None:
            self._sig = self.unsig[:, 0].dot(self.vfs)
            self._sig += self.unsig[:, 1].dot(self.ofs)
            for control in self.ctrls:
                if control in self.sys.ctrls:
                    ctrl = self.ctrls[control]
                    ctrlrad = radians(ctrl)
                    index = self.sys.ctrls[control]
                    if ctrl >= 0.0:
                        indv = index[0]
                        indo = index[1]
                    else:
                        indv = index[2]
                        indo = index[3]
                    self._sig += ctrlrad*(self.unsig[:, indv].dot(self.vfs))
                    self._sig += ctrlrad*(self.unsig[:, indo].dot(self.ofs))
        return self._sig

    def calc_sigg(self, sig: 'NDArray') -> 'NDArray':
        return self.sys.edges_array @ sig

    @property
    def sigg(self) -> 'NDArray':
        if self._sigg is None:
            self._sigg = self.calc_sigg(self.sig)
        return self._sigg

    @property
    def mu(self) -> 'NDArray':
        if self._mu is None:
            self._mu = self.unmu[:, 0].dot(self.vfs)
            self._mu += self.unmu[:, 1].dot(self.ofs)
            for control in self.ctrls:
                if control in self.sys.ctrls:
                    ctrl = self.ctrls[control]
                    ctrlrad = radians(ctrl)
                    index = self.sys.ctrls[control]
                    if ctrl >= 0.0:
                        indv = index[0]
                        indo = index[1]
                    else:
                        indv = index[2]
                        indo = index[3]
                    self._mu += ctrlrad*(self.unmu[:, indv].dot(self.vfs))
                    self._mu += ctrlrad*(self.unmu[:, indo].dot(self.ofs))
        return self._mu

    def calc_mug(self, mu: 'NDArray') -> 'NDArray':
        return self.sys.edges_array @ mu

    @property
    def mug(self):
        if self._mug is None:
            self._mug = self.calc_mug(self.mu)
        return self._mug

    @property
    def phi(self):
        if self._phi is None:
            self._phi = self.unphi[:, 0].dot(self.vfs)
            self._phi += self.unphi[:, 1].dot(self.ofs)
            for control in self.ctrls:
                if control in self.sys.ctrls:
                    ctrl = self.ctrls[control]
                    ctrlrad = radians(ctrl)
                    index = self.sys.ctrls[control]
                    if ctrl >= 0.0:
                        indv = index[0]
                        indo = index[1]
                    else:
                        indv = index[2]
                        indo = index[3]
                    self._phi += ctrlrad*(self.unphi[:, indv].dot(self.vfs))
                    self._phi += ctrlrad*(self.unphi[:, indo].dot(self.ofs))
        return self._phi

    @property
    def nvg(self):
        if self._nvg is None:
            self._nvg = self.unnvg[:, 0].dot(self.vfs)
            self._nvg += self.unnvg[:, 1].dot(self.ofs)
            for control in self.ctrls:
                if control in self.sys.ctrls:
                    ctrl = self.ctrls[control]
                    ctrlrad = radians(ctrl)
                    index = self.sys.ctrls[control]
                    if ctrl >= 0.0:
                        indv = index[0]
                        indo = index[1]
                    else:
                        indv = index[2]
                        indo = index[3]
                    self._nvg += ctrlrad*(self.unnvg[:, indv].dot(self.vfs))
                    self._nvg += ctrlrad*(self.unnvg[:, indo].dot(self.ofs))
        return self._nvg

    # def calc_qloc_old(self, mu: 'NDArray', vfs: Vector | None = None,
    #                   ofs: Vector | None = None) -> Vector2D:
    #     vfsg = Vector.zeros(self.arm.shape)
    #     if vfs is not None:
    #         vfsg += vfs
    #     if ofs is not None:
    #         vfsg += self.arm.cross(ofs)
    #     vl = self.sys.pnldirx.dot(vfsg)
    #     vt = self.sys.pnldiry.dot(vfsg)
    #     ql = zeros(self.sys.numpnl)
    #     qt = zeros(self.sys.numpnl)
    #     for pnl in self.sys.pnls.values():
    #         ql[pnl.ind], qt[pnl.ind] = pnl.diff_mu_old(mu)
    #     # print(f'{ql = }')
    #     # print(f'{qt = }')
    #     return Vector2D(vl + ql, vt + qt)

    def calc_qloc(self, mu: 'NDArray', mug: 'NDArray | None' = None, *,
                  vfs: Vector | None = None,
                  ofs: Vector | None = None) -> Vector2D:
        if mug is None:
            mug = self.calc_mug(mu)
        vfsg = Vector.zeros(self.arm.shape)
        if vfs is not None:
            vfsg += vfs
        if ofs is not None:
            vfsg += self.arm.cross(ofs)
        vfsl_2d = Vector2D.zeros(self.sys.numpnl)
        diff_mu = Vector2D.zeros(self.sys.numpnl)
        for pnl in self.sys.pnls.values():
            vel_fs = pnl.crd.vector_to_local(vfsg[pnl.ind])
            vfsl_2d[pnl.ind] = Vector2D.from_obj(vel_fs)
            diff_mu[pnl.ind] = pnl.diff_mu(mu, mug)
        # print(f'{diff_mu.x = }')
        # print(f'{diff_mu.y = }')
        return vfsl_2d - diff_mu

    @property
    def qloc(self) -> Vector2D:
        if self._qloc is None:
            self._qloc = self.calc_qloc(self.mu, self.mug,
                                        vfs = self.vfs, ofs = self.ofs)
            # self._qloc = self.calc_qloc_old(self.mu, vfs = self.vfs,
            #                                 ofs = self.ofs)
        return self._qloc

    @property
    def qs(self) -> float:
        if self._qs is None:
            self._qs = self.qloc.return_magnitude()
        return self._qs

    @property
    def cp(self) -> float:
        if self._cp is None:
            self._cp = 1.0 - square(self.qs)/self.speed**2
        return self._cp

    @property
    def nfres(self) -> 'NearFieldResult':
        if self._nfres is None:
            self._nfres = NearFieldResult(self, self.cp)
        return self._nfres

    @property
    def strpres(self) -> 'StripResult':
        if self._strpres is None:
            if self.sys.srfcs is not None:
                self._strpres = StripResult(self.nfres)
        return self._strpres

    @property
    def ffres(self) -> 'FarFieldResult':
        if self._ffres is None:
            if self.sys.srfcs is not None:
                self._ffres = FarFieldResult(self)
        return self._ffres

    @property
    def stres(self) -> 'StabilityResult':
        if self._stres is None:
            self._stres = StabilityResult(self)
        return self._stres

    def gctrlp_single(self, control: str):
        indv = self.sys.ctrls[control][0]
        indo = self.sys.ctrls[control][1]
        dmu = self.unmu[:, indv].dot(self.vfs) + self.unmu[:, indo].dot(self.ofs)
        dqloc = self.calc_qloc(dmu)#, vfs=self.vfs, ofs=self.ofs)
        # qloc = self.calc_qloc_old(mu, vfs=self.vfs, ofs=self.ofs)
        vdot = self.qloc.dot(dqloc)
        return (-2.0/self.speed**2)*vdot

    def gctrln_single(self, control: str):
        indv = self.sys.ctrls[control][2]
        indo = self.sys.ctrls[control][3]
        dmu = self.unmu[:, indv].dot(self.vfs) + self.unmu[:, indo].dot(self.ofs)
        dqloc = self.calc_qloc(dmu)#, vfs=self.vfs, ofs=self.ofs)
        # qloc = self.calc_qloc_old(mu, vfs=self.vfs, ofs=self.ofs)
        vdot = self.qloc.dot(dqloc)
        return (-2.0/self.speed**2)*vdot

    @property
    def ctresp(self):
        if self._ctresp is None:
            self._ctresp = {}
            for control in self.sys.ctrls:
                dcp = self.gctrlp_single(control)
                self._ctresp[control] = NearFieldResult(self, dcp)
        return self._ctresp

    @property
    def ctresn(self):
        if self._ctresn is None:
            self._ctresn = {}
            for control in self.sys.ctrls:
                dcp = self.gctrln_single(control)
                self._ctresn[control] = NearFieldResult(self, dcp)
        return self._ctresn

    def plot_strip_lift_force_distribution(self, ax=None, axis: str='b',
                                           surfaces: list=None, normalise: bool=False,
                                           label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    l = [self.strpres.lift[strp.ind]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    l = [self.strpres.lift[strp.ind]/strp.width for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, l, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, l, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(l, z, label=thislabel)
            ax.legend()
        return ax

    def plot_strip_side_force_distribution(self, ax=None, axis: str='b',
                                           surfaces: list=None, normalise: bool=False,
                                           label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    f = [self.strpres.side[strp.ind]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    f = [self.strpres.side[strp.ind]/strp.width for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, f, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, f, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(f, z, label=thislabel)
            ax.legend()
        return ax

    def plot_strip_drag_force_distribution(self, ax=None, axis: str='b',
                                           surfaces: list=None, normalise: bool=False,
                                           label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    d = [self.strpres.drag[strp.ind]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    d = [self.strpres.drag[strp.ind]/strp.width for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, d, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, d, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(d, z, label=thislabel)
            ax.legend()
        return ax



    def plot_strip_pitch_moment_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False,
                                             label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    l = [self.strpres.stmom.y[strp.ind]/strp.area/strp.chord/self.qfs for strp in srfc.strps]
                else:
                    l = [self.strpres.stmom.y[strp.ind]/strp.width/strp.chord for strp in srfc.strps]
                if label is None:
                    thislabel = self.name + ' for ' + srfc.name
                else:
                    if not onesrfc:
                        thislabel = label + ' for ' + srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, l, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, l, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(l, z, label=thislabel)
            ax.legend()
        return ax

    def plot_trefftz_lift_force_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False,
                                             label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    l = [self.ffres.lift[strp.ind]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    l = [self.ffres.lift[strp.ind]/strp.width for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, l, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, l, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(l, z, label=thislabel)
            ax.legend()
        return ax

    def plot_trefftz_side_force_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False,
                                             label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    f = [self.ffres.side[strp.ind]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    f = [self.ffres.side[strp.ind]/strp.width for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, f, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, f, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(f, z, label=thislabel)
            ax.legend()
        return ax

    def plot_trefftz_drag_force_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False,
                                             label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                if normalise:
                    d = [self.ffres.drag[strp.ind]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    d = [self.ffres.drag[strp.ind]/strp.width for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, d, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, d, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(d, z, label=thislabel)
            ax.legend()
        return ax

    def plot_trefftz_wash_distribution(self, ax=None, axis: str='b',
                                       surfaces: list=None, label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                w = [self.ffres.wash[strp.ind] for strp in srfc.strps]
                # wa = [self.ffres.washa[strp.ind] for strp in srfc.strps]
                # wb = [self.ffres.washb[strp.ind] for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, w, label=thislabel)
                        # ax.plot(b, wa, label=thislabel)
                        # ax.plot(b, wb, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, w, label=thislabel)
                        # ax.plot(y, wa, label=thislabel)
                        # ax.plot(y, wb, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(w, z, label=thislabel)
                        # ax.plot(wa, z, label=thislabel)
                        # ax.plot(wb, z, label=thislabel)
            ax.legend()
        return ax

    def plot_trefftz_circulation_distribution(self, ax=None, axis: str='b',
                                              surfaces: list=None, label: str=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = self.sys.srfcs
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            onesrfc = len(srfcs) == 1
            for srfc in srfcs:
                c = [self.ffres.circ[strp.ind] for strp in srfc.strps]
                # ma = [self.ffres.mua[strp.ind] for strp in srfc.strps]
                # mb = [self.ffres.mub[strp.ind] for strp in srfc.strps]
                if label is None:
                    thislabel = self.name+' for '+srfc.name
                else:
                    if not onesrfc:
                        thislabel = label+' for '+srfc.name
                    else:
                        thislabel = label
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, c, label=thislabel)
                        # ax.plot(b, ma, label=thislabel)
                        # ax.plot(b, mb, label=thislabel)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, c, label=thislabel)
                        # ax.plot(y, ma, label=thislabel)
                        # ax.plot(y, mb, label=thislabel)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(c, z, label=thislabel)
                        # ax.plot(ma, z, label=thislabel)
                        # ax.plot(mb, z, label=thislabel)
            ax.legend()
        return ax

    def to_result(self, name: str=''):
        if name == '':
            name = self.name
        res = PanelResult(name, self.sys)
        res.set_density(rho=self.rho)
        res.set_state(speed=self.speed, alpha=self.alpha, beta=self.beta,
                      pbo2v=self.pbo2v, qco2v=self.qco2v, rbo2v=self.rbo2v)
        res.set_controls(**self.ctrls)
        res.set_cg(self.rcg)
        return res

    @property
    def stability_derivatives(self):
        return self.stres.stability_derivatives

    @property
    def stability_derivatives_body(self):
        return self.stres.stability_derivatives_body

    @property
    def control_derivatives(self):
        from . import sfrm
        report = MDReport()
        heading = MDHeading('Control Derivatives', 2)
        report.add_object(heading)
        for control in self.ctrls:
            letter = control[0]
            heading = MDHeading(f'{control.capitalize()} Derivatives', 3)
            report.add_object(heading)
            table = MDTable()
            table.add_column(f'CLd{letter:s}', sfrm)
            table.add_column(f'CYd{letter:s}', sfrm)
            table.add_column(f'Cld{letter:s}', sfrm)
            table.add_column(f'Cmd{letter:s}', sfrm)
            table.add_column(f'Cnd{letter:s}', sfrm)
            if self.ctrls[control] >= 0.0:
                ctresp = self.ctresp[control]
                table.add_row([ctresp.CL, ctresp.CY, ctresp.Cl, ctresp.Cm, ctresp.Cn])
            if self.ctrls[control] <= 0.0:
                ctresp = self.ctresp[control]
                table.add_row([ctresp.CL, ctresp.CY, ctresp.Cl, ctresp.Cm, ctresp.Cn])
            report.add_object(table)
        return report

    @property
    def surface_loads(self):
        if self.sys.srfcs is not None:
            report = MDReport()
            heading = MDHeading('Surface Loads', 2)
            report.add_object(heading)
            table = MDTable()
            table.add_column('xref', '.3f', data=[self.rcg.x])
            table.add_column('yref', '.3f', data=[self.rcg.y])
            table.add_column('zref', '.3f', data=[self.rcg.z])
            report.add_object(table)
            table1 = MDTable()
            table1.add_column('Name', 's')
            table1.add_column('Fx', '.3f')
            table1.add_column('Fy', '.3f')
            table1.add_column('Fz', '.3f')
            table1.add_column('Mx', '.3f')
            table1.add_column('My', '.3f')
            table1.add_column('Mz', '.3f')
            table2 = MDTable()
            table2.add_column('Name', 's')
            table2.add_column('Area', '.3f')
            table2.add_column('Di', '.3f')
            table2.add_column('Y', '.3f')
            table2.add_column('L', '.3f')
            table2.add_column('CDi', '.7f')
            table2.add_column('CY', '.5f')
            table2.add_column('CL', '.5f')
            Ditot = 0.0
            Ytot = 0.0
            Ltot = 0.0
            for srfc in self.sys.srfcs:
                area = srfc.area
                ind = srfc.pinds
                frc = self.nfres.nffrc[ind].sum()
                mom = self.nfres.nfmom[ind].sum()
                table1.add_row([srfc.name, frc.x, frc.y, frc.z, mom.x, mom.y, mom.z])
                if area > 0.0:
                    Di = frc.dot(self.acs.dirx)
                    Y = frc.dot(self.acs.diry)
                    L = frc.dot(self.acs.dirz)
                    CDi = Di/self.qfs/area
                    CY = Y/self.qfs/area
                    CL = L/self.qfs/area
                    table2.add_row([srfc.name, area, Di, Y, L, CDi, CY, CL])
                    Ditot += Di
                    Ytot += Y
                    Ltot += L
            frc = self.nfres.nffrc.sum()
            mom = self.nfres.nfmom.sum()
            table1.add_row(['Total', frc.x, frc.y, frc.z, mom.x, mom.y, mom.z])
            report.add_object(table1)
            table = MDTable()
            table.add_column('Density', '.3f', data=[self.rho])
            table.add_column('Speed', '.3f', data=[self.speed])
            table.add_column('Dynamic Pressure', '.1f', data=[self.qfs])
            report.add_object(table)
            table2.add_row(['Total', self.sys.sref, Ditot, Ytot, Ltot,
                            self.nfres.CDi, self.nfres.CY, self.nfres.CL])
            report.add_object(table2)
            return report

    def to_mdobj(self) -> MDReport:
        from . import cfrm, dfrm, efrm

        report = MDReport()
        heading = MDHeading(f'Panel Result {self.name} for {self.sys.name}', 1)
        report.add_object(heading)
        table = MDTable()
        table.add_column('Alpha (deg)', cfrm, data=[self.alpha])
        table.add_column('Beta (deg)', cfrm, data=[self.beta])
        table.add_column('Speed', cfrm, data=[self.speed])
        table.add_column('Rho', cfrm, data=[self.rho])
        table.add_column('Mach', efrm, data=[self.mach])
        report.add_object(table)
        table = MDTable()
        table.add_column('pb/2V (rad)', cfrm, data=[self.pbo2v])
        table.add_column('qc/2V (rad)', cfrm, data=[self.qco2v])
        table.add_column('rb/2V (rad)', cfrm, data=[self.rbo2v])
        report.add_object(table)
        table = MDTable()
        table.add_column('xcg', '.5f', data=[self.rcg.x])
        table.add_column('ycg', '.5f', data=[self.rcg.y])
        table.add_column('zcg', '.5f', data=[self.rcg.z])
        report.add_object(table)
        if len(self.ctrls) > 0:
            table = MDTable()
            for control in self.ctrls:
                ctrl = self.ctrls[control]
                control = control.capitalize()
                table.add_column(f'{control} (deg)', cfrm, data=[ctrl])
            report.add_object(table)
        # if self.sys.cdo != 0.0:
        #     table = MDTable()
        #     table.add_column('CDo', dfrm, data=[self.pdres.CDo])
        #     table.add_column('CYo', cfrm, data=[self.pdres.CY])
        #     table.add_column('CLo', cfrm, data=[self.pdres.CL])
        #     table.add_column('Clo', cfrm, data=[self.pdres.Cl])
        #     table.add_column('Cmo', cfrm, data=[self.pdres.Cm])
        #     table.add_column('Cno', cfrm, data=[self.pdres.Cn])
        #     outstr += table._repr_markdown_()
        if self.nfres is not None:
            table = MDTable()
            table.add_column('Cx', cfrm, data=[self.nfres.Cx])
            table.add_column('Cy', cfrm, data=[self.nfres.Cy])
            table.add_column('Cz', cfrm, data=[self.nfres.Cz])
            report.add_object(table)
            table = MDTable()
            table.add_column('CDi', dfrm, data=[self.nfres.CDi])
            table.add_column('CY', cfrm, data=[self.nfres.CY])
            table.add_column('CL', cfrm, data=[self.nfres.CL])
            table.add_column('Cl', cfrm, data=[self.nfres.Cl])
            table.add_column('Cm', cfrm, data=[self.nfres.Cm])
            table.add_column('Cn', cfrm, data=[self.nfres.Cn])
            # if self.sys.cdo != 0.0:
            #     lod = self.nfres.CL/(self.pdres.CDo+self.nfres.CDi)
            #     table.add_column('L/D', '.5g', data=[lod])
            report.add_object(table)
        if self.ffres is not None:
            table = MDTable()
            table.add_column('CDi_ff', dfrm, data=[self.ffres.CDi])
            table.add_column('CY_ff', cfrm, data=[self.ffres.CY])
            table.add_column('CL_ff', cfrm, data=[self.ffres.CL])
            table.add_column('e', efrm, data=[self.ffres.e])
            # if self.sys.cdo != 0.0:
            #     lod_ff = self.ffres.CL/(self.pdres.CDo+self.ffres.CDi)
            #     table.add_column('L/D_ff', '.5g', data=[lod_ff])
            report.add_object(table)

        return report

    @classmethod
    def from_dict(cls, sys: 'System', resdata: dict[str, Any]) -> 'PanelResult':
        name = resdata['name']
        if 'inherit' in resdata:
            inherit = resdata['inherit']
            if inherit in sys.results:
                pres = sys.results[inherit].to_result(name=name)
        else:
            pres = PanelResult(name, sys)
        for key in resdata:
            if key == 'name':
                continue
            elif key == 'inherit':
                continue
            elif key == 'density':
                rho = resdata['density']
                pres.set_density(rho=rho)
            elif key == 'mach':
                mach = resdata['mach']
                pres.set_state(mach=mach)
            elif key == 'speed':
                speed = resdata['speed']
                pres.set_state(speed=speed)
            elif key ==  'alpha':
                alpha = resdata['alpha']
                pres.set_state(alpha=alpha)
            elif key ==  'beta':
                beta = resdata['beta']
                pres.set_state(beta=beta)
            elif key ==  'pbo2v':
                pbo2v = resdata['pbo2v']
                pres.set_state(pbo2v=pbo2v)
            elif key ==  'qco2v':
                qco2v = resdata['qco2v']
                pres.set_state(qco2v=qco2v)
            elif key ==  'rbo2v':
                rbo2v = resdata['rbo2v']
                pres.set_state(rbo2v=rbo2v)
            elif key in pres.ctrls:
                pres.ctrls[key] = resdata[key]
            elif key == 'rcg':
                rcgdata = resdata[key]
                rcg = Vector(rcgdata['x'], rcgdata['y'], rcgdata['z'])
                pres.set_cg(rcg)
        sys.results[name] = pres
        return pres

    def __str__(self) -> str:
        return self.to_mdobj().__str__()

    def __repr__(self):
        return f'<PanelResult: {self.name}>'

    def _repr_markdown_(self):
        return self.to_mdobj()._repr_markdown_()

def trig_angle(angle: float):
    '''Calculates cos(angle) and sin(angle) with angle in degrees.'''
    angrad = radians(angle)
    cosang = cos(angrad)
    sinang = sin(angrad)
    return cosang, sinang


class NearFieldResult():
    res: PanelResult = None
    nfcp: 'NDArray' = None
    _nfprs: 'NDArray' = None
    _nffrc: Vector = None
    _nfmom: Vector = None
    _nffrctot: Vector = None
    _nfmomtot: Vector = None
    _Cx: float = None
    _Cy: float = None
    _Cz: float = None
    _Cmx: float = None
    _Cmy: float = None
    _Cmz: float = None
    _CDi: float = None
    _CY: float = None
    _CL: float = None
    _e: float = None
    _Cl: float = None
    _Cm: float = None
    _Cn: float = None

    def __init__(self, res: PanelResult, nfcp: 'NDArray'):
        self.res = res
        self.nfcp = nfcp

    @property
    def nfprs(self) -> 'NDArray':
        if self._nfprs is None:
            self._nfprs = self.res.qfs*self.nfcp
        return self._nfprs

    @property
    def nffrc(self) -> Vector:
        if self._nffrc is None:
            nrms = self.res.sys.nrms
            pnla = self.res.sys.pnla
            self._nffrc = -nrms*self.nfprs*pnla
        return self._nffrc

    @property
    def nfmom(self) -> Vector:
        if self._nfmom is None:
            self._nfmom = self.res.arm.cross(self.nffrc)
        return self._nfmom

    @property
    def nffrctot(self) -> Vector:
        if self._nffrctot is None:
            self._nffrctot = self.nffrc.sum()
        return self._nffrctot

    @property
    def nfmomtot(self) -> Vector:
        if self._nfmomtot is None:
            self._nfmomtot = self.nfmom.sum()
        return self._nfmomtot

    @property
    def Cx(self) -> float:
        if self._Cx is None:
            self._Cx = self.nffrctot.x/self.res.qfs/self.res.sys.sref
            self._Cx = fix_zero(self._Cx)
        return self._Cx

    @property
    def Cy(self) -> float:
        if self._Cy is None:
            self._Cy = self.nffrctot.y/self.res.qfs/self.res.sys.sref
            self._Cy = fix_zero(self._Cy)
        return self._Cy

    @property
    def Cz(self) -> float:
        if self._Cz is None:
            self._Cz = self.nffrctot.z/self.res.qfs/self.res.sys.sref
            self._Cz = fix_zero(self._Cz)
        return self._Cz

    @property
    def Cmx(self) -> float:
        if self._Cmx is None:
            self._Cmx = self.nfmomtot.x/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cmx = fix_zero(self._Cmx)
        return self._Cmx

    @property
    def Cmy(self) -> float:
        if self._Cmy is None:
            self._Cmy = self.nfmomtot.y/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cmy = fix_zero(self._Cmy)
        return self._Cmy

    @property
    def Cmz(self) -> float:
        if self._Cmz is None:
            self._Cmz = self.nfmomtot.z/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cmz = fix_zero(self._Cmz)
        return self._Cmz

    @property
    def CDi(self) -> float:
        if self._CDi is None:
            Di = self.nffrctot.dot(self.res.acs.dirx)
            self._CDi = Di/self.res.qfs/self.res.sys.sref
            self._CDi = fix_zero(self._CDi)
        return self._CDi

    @property
    def CY(self) -> float:
        if self._CY is None:
            Y = self.nffrctot.dot(self.res.acs.diry)
            self._CY = Y/self.res.qfs/self.res.sys.sref
            self._CY = fix_zero(self._CY)
        return self._CY

    @property
    def CL(self) -> float:
        if self._CL is None:
            L = self.nffrctot.dot(self.res.acs.dirz)
            self._CL = L/self.res.qfs/self.res.sys.sref
            self._CL = fix_zero(self._CL)
        return self._CL

    @property
    def Cl(self) -> float:
        if self._Cl is None:
            l = self.nfmomtot.dot(self.res.scs.dirx)
            self._Cl = l/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cl = fix_zero(self._Cl)
        return self._Cl

    @property
    def Cm(self) -> float:
        if self._Cm is None:
            m = self.nfmomtot.dot(self.res.scs.diry)
            self._Cm = m/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cm = fix_zero(self._Cm)
        return self._Cm

    @property
    def Cn(self) -> float:
        if self._Cn is None:
            n = self.nfmomtot.dot(self.res.scs.dirz)
            self._Cn = n/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cn = fix_zero(self._Cn)
        return self._Cn

    @property
    def e(self) -> float:
        if self._e is None:
            if self.CDi <= 0.0:
                self._e = float('nan')
            elif self.CL == 0.0 and self.CY == 0.0:
                self._e = 0.0
            else:
                self._e = (self.CL**2+self.CY**2)/pi/self.res.sys.ar/self.CDi
                self._e = fix_zero(self._e)
        return self._e


class StabilityNearFieldResult():
    res: PanelResult = None
    dvfs: Vector = None
    dofs: Vector = None
    dacs: dict[str, Vector] = None
    dscs: dict[str, Vector] = None
    _dmu: 'NDArray' = None
    _dqloc: Vector2D = None
    _vdot: 'NDArray' = None
    _dvdot: 'NDArray' = None
    _dspeed: 'NDArray' = None
    _dcp: 'NDArray' = None
    _dqfs: 'NDArray' = None
    _dprs: 'NDArray' = None
    _dfrc: Vector = None
    _dmom: Vector = None
    _dfrctot: Vector = None
    _dmomtot: Vector = None
    _Cx: float = None
    _Cy: float = None
    _Cz: float = None
    _Cmx: float = None
    _Cmy: float = None
    _Cmz: float = None
    _CDi: float = None
    _CY: float = None
    _CL: float = None
    _e: float = None
    _Cl: float = None
    _Cm: float = None
    _Cn: float = None

    def __init__(self, res: PanelResult, dvfs: Vector | None = None, dofs: Vector | None = None,
                 dacs: dict[str, Vector] = None, dscs: dict[str, Vector] | None = None) -> None:
        self.res = res
        self.dvfs = dvfs
        self.dofs = dofs
        self.dacs = dacs
        self.dscs = dscs

    @property
    def dmu(self) -> 'NDArray':
        if self._dmu is None:
            self._dmu = zeros(len(self.res.sys.pnls))
            if self.dvfs is not None:
                self._dmu += self.res.unmu[:, 0].dot(self.dvfs)
            if self.dofs is not None:
                self._dmu += self.res.unmu[:, 1].dot(self.dofs)
        return self._dmu

    @property
    def dqloc(self) -> Vector2D:
        if self._dqloc is None:
            self._dqloc = self.res.calc_qloc(self.dmu, vfs=self.dvfs, ofs=self.dofs)
            # self._dqloc = self.res.calc_qloc_old(self.dmu, self.dvfs, self.dofs)
        return self._dqloc

    @property
    def vdot(self) -> 'NDArray':
        if self._vdot is None:
            self._vdot = self.res.qloc.dot(self.res.qloc)
        return self._vdot

    @property
    def dvdot(self) -> 'NDArray':
        if self._dvdot is None:
            self._dvdot = self.res.qloc.dot(self.dqloc)
        return self._dvdot

    @property
    def dspeed(self) -> 'NDArray':
        if self._dspeed is None:
            if self.dvfs is not None:
                self._dspeed = self.dvfs.return_magnitude()
            else:
                self._dspeed = 0.0
        return self._dspeed

    @property
    def dcp(self) -> 'NDArray':
        if self._dcp is None:
            # self._dcp = -2*self.dvdot/self.res.speed**2 # Needs fixing using Chain Rule
            self._dcp = 2*(self.vdot*self.dspeed - self.dvdot*self.res.speed)/self.res.speed**3
        return self._dcp

    @property
    def dqfs(self) -> 'NDArray':
        if self._dqfs is None:
            self._dqfs = self.res.rho*self.res.speed*self.dspeed
        return self._dqfs

    @property
    def dprs(self) -> 'NDArray':
        if self._dprs is None:
            self._dprs = self.res.qfs*self.dcp + self.dqfs*self.res.cp
        return self._dprs

    @property
    def dfrc(self) -> Vector:
        if self._dfrc is None:
            nrms = self.res.sys.nrms
            pnla = self.res.sys.pnla
            self._dfrc = -nrms*self.dprs*pnla
        return self._dfrc

    @property
    def dmom(self) -> Vector:
        if self._dmom is None:
            self._dmom = self.res.arm.cross(self.dfrc)
        return self._dmom

    @property
    def dfrctot(self) -> Vector:
        if self._dfrctot is None:
            self._dfrctot = self.dfrc.sum()
        return self._dfrctot

    @property
    def dmomtot(self) -> Vector:
        if self._dmomtot is None:
            self._dmomtot = self.dmom.sum()
        return self._dmomtot

    @property
    def Cx(self) -> float:
        if self._Cx is None:
            self._Cx = self.dfrctot.x/self.res.qfs/self.res.sys.sref
            self._Cx = fix_zero(self._Cx)
        return self._Cx

    @property
    def Cy(self) -> float:
        if self._Cy is None:
            self._Cy = self.dfrctot.y/self.res.qfs/self.res.sys.sref
            self._Cy = fix_zero(self._Cy)
        return self._Cy

    @property
    def Cz(self) -> float:
        if self._Cz is None:
            self._Cz = self.dfrctot.z/self.res.qfs/self.res.sys.sref
            self._Cz = fix_zero(self._Cz)
        return self._Cz

    @property
    def Cmx(self) -> float:
        if self._Cmx is None:
            self._Cmx = self.dmomtot.x/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cmx = fix_zero(self._Cmx)
        return self._Cmx

    @property
    def Cmy(self) -> float:
        if self._Cmy is None:
            self._Cmy = self.dmomtot.y/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cmy = fix_zero(self._Cmy)
        return self._Cmy

    @property
    def Cmz(self) -> float:
        if self._Cmz is None:
            self._Cmz = self.dmomtot.z/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cmz = fix_zero(self._Cmz)
        return self._Cmz

    @property
    def CDi(self) -> float:
        if self._CDi is None:
            Di = self.dfrctot.dot(self.res.acs.dirx)
            if self.dacs is not None:
                Di += self.res.nfres.nffrctot.dot(self.dacs['x'])
            self._CDi = Di/self.res.qfs/self.res.sys.sref
            self._CDi = fix_zero(self._CDi)
        return self._CDi

    @property
    def CY(self) -> float:
        if self._CY is None:
            Y = self.dfrctot.dot(self.res.acs.diry)
            if self.dacs is not None:
                Y += self.res.nfres.nffrctot.dot(self.dacs['y'])
            self._CY = Y/self.res.qfs/self.res.sys.sref
            self._CY = fix_zero(self._CY)
        return self._CY

    @property
    def CL(self) -> float:
        if self._CL is None:
            L = self.dfrctot.dot(self.res.acs.dirz)
            if self.dacs is not None:
                L += self.res.nfres.nffrctot.dot(self.dacs['z'])
            self._CL = L/self.res.qfs/self.res.sys.sref
            self._CL = fix_zero(self._CL)
        return self._CL

    @property
    def Cl(self) -> float:
        if self._Cl is None:
            l = self.dmomtot.dot(self.res.scs.dirx)
            if self.dscs is not None:
                l += self.res.nfres.nfmomtot.dot(self.dscs['x'])
            self._Cl = l/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cl = fix_zero(self._Cl)
        return self._Cl

    @property
    def Cm(self) -> float:
        if self._Cm is None:
            m = self.dmomtot.dot(self.res.scs.diry)
            if self.dscs is not None:
                m += self.res.nfres.nfmomtot.dot(self.dscs['y'])
            self._Cm = m/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cm = fix_zero(self._Cm)
        return self._Cm

    @property
    def Cn(self) -> float:
        if self._Cn is None:
            n = self.dmomtot.dot(self.res.scs.dirz)
            if self.dscs is not None:
                n += self.res.nfres.nfmomtot.dot(self.dscs['z'])
            self._Cn = n/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cn = fix_zero(self._Cn)
        return self._Cn


class StripResult():
    nfres: 'NearFieldResult' = None
    _stfrc: Vector = None
    _stmom: Vector = None
    _stcp = None
    _lift = None
    _side = None
    _drag = None
    _cp = None

    def __init__(self, nfres: NearFieldResult) -> None:
        self.nfres = nfres

    @property
    def stfrc(self):
        if self._stfrc is None:
            sys = self.nfres.res.sys
            num = len(sys.strps)
            self._stfrc = Vector.zeros(num)
            for strp in sys.strps:
                i = strp.ind
                for pnl in strp.pnls:
                    j = pnl.ind
                    self._stfrc[i] += self.nfres.nffrc[j]
        return self._stfrc

    @property
    def stmom(self):
        if self._stmom is None:
            sys = self.nfres.res.sys
            num = len(sys.strps)
            self._stmom = Vector.zeros(num)
            for strp in sys.strps:
                i = strp.ind
                for pnl in strp.pnls:
                    j = pnl.ind
                    rref = pnl.pnto - strp.point
                    self._stmom[i] += rref.cross(self.nfres.nffrc[j])
        return self._stmom

    @property
    def stcp(self):
        if self._stcp is None:
            sys = self.nfres.res.sys
            num = len(sys.strps)
            self._stcp = zeros(num)
            for strp in sys.strps:
                i = strp.ind
                if abs(self.stfrc[i].z) > 1e-12:
                    self._stcp[i] = -self.stmom[i].y/self.stfrc[i].z/strp.chord
        return self._stcp

    @property
    def drag(self):
        if self._drag is None:
            self._drag = zeros(self.stfrc.size)
            for i in range(self.stfrc.size):
                self._drag[i] = self.nfres.res.acs.dirx.dot(self.stfrc[i])
        return self._drag

    @property
    def side(self):
        if self._side is None:
            self._side = zeros(self.stfrc.size)
            for i in range(self.stfrc.size):
                self._side[i] = self.nfres.res.acs.diry.dot(self.stfrc[i])
        return self._side

    @property
    def lift(self):
        if self._lift is None:
            self._lift = zeros(self.stfrc.size)
            for i in range(self.stfrc.size):
                self._lift[i] = self.nfres.res.acs.dirz.dot(self.stfrc[i])
        return self._lift

    def to_mdobj(self) -> MDReport:

        res = self.nfres.res
        resname = res.name
        sys = res.sys
        sysname = sys.name
        ind = [strp.ind for strp in sys.strps]
        pntx = [strp.point.x for strp in sys.strps]
        pnty = [strp.point.y for strp in sys.strps]
        pntz = [strp.point.z for strp in sys.strps]
        frcx = self.stfrc.x[ind].transpose().tolist()[0]
        frcy = self.stfrc.y[ind].transpose().tolist()[0]
        frcz = self.stfrc.z[ind].transpose().tolist()[0]
        momx = self.stmom.x[ind].transpose().tolist()[0]
        momy = self.stmom.y[ind].transpose().tolist()[0]
        momz = self.stmom.z[ind].transpose().tolist()[0]

        report = MDReport()
        heading = MDHeading(f'Strip Results {resname} for {sysname}', 1)
        report.add_object(heading)
        table = MDTable()
        table.add_column('Strip', 'd', data=ind)
        table.add_column('Point X', '.3f', data=pntx)
        table.add_column('Point Y', '.3f', data=pnty)
        table.add_column('Point X', '.3f', data=pntz)
        table.add_column('Force X', '.3f', data=frcx)
        table.add_column('Force Y', '.3f', data=frcy)
        table.add_column('Force X', '.3f', data=frcz)
        table.add_column('Moment X', '.3f', data=momx)
        table.add_column('Moment Y', '.3f', data=momy)
        table.add_column('Moment Z', '.3f', data=momz)
        report.add_object(table)

        return report

    def __repr__(self):
        return f'<StripResult: {self.nfres.res.name}>'

    def __str__(self) -> str:
        return self.to_mdobj().__str__()

    def _repr_markdown_(self):
        return self.to_mdobj()._repr_markdown_()

class FarFieldResult():
    res = None
    _ffmu = None
    _ffwsh = None
    _fffrc = None
    _ffmom = None
    _fffrctot = None
    _ffmomtot = None
    _circ = None
    _wash = None
    _drag = None
    _side = None
    _lift = None
    _CDi = None
    _CY = None
    _CL = None
    _Cl = None
    _Cm = None
    _Cn = None
    _e = None
    _lod = None

    def __init__(self, res: PanelResult):
        self.res = res

    @property
    def ffmu(self):
        if self._ffmu is None:
            numhsv = self.res.sys.numhsv
            self._ffmu = zeros(numhsv)
            for ind, hsv in enumerate(self.res.sys.hsvs):
                self._ffmu[ind] = self.res.mu[hsv.ind]
        return self._ffmu

    @property
    def ffwsh(self):
        if self._ffwsh is None:
            self._ffwsh = self.res.sys.awh@self.ffmu
        return self._ffwsh

    @property
    def fffrc(self):
        if self._fffrc is None:
            x = self.res.rho*self.ffmu*self.res.sys.adh@self.ffmu
            y = self.res.rho*self.res.speed*self.ffmu*self.res.sys.ash
            z = self.res.rho*self.res.speed*self.ffmu*self.res.sys.alh
            self._fffrc = Vector(x, y, z)
        return self._fffrc

    @property
    def ffmom(self):
        if self._ffmom is None:
            self._ffmom = self.res.brm.cross(self.fffrc)
        return self._ffmom

    @property
    def fffrctot(self):
        if self._fffrctot is None:
            self._fffrctot = self.fffrc.sum()
        return self._fffrctot

    @property
    def ffmomtot(self):
        if self._ffmomtot is None:
            self._ffmomtot = self.ffmom.sum()
        return self._ffmomtot

    @property
    def circ(self):
        if self._circ is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._circ = zeros((num, 1))
            for i, strp in enumerate(sys.strps):
                if not strp.nohsv:
                    pnla = strp.pnls[0]
                    pnlb = strp.pnls[-1]
                    mua = self.res.mu[pnla.ind]
                    mub = self.res.mu[pnlb.ind]
                    self._circ[i] = mua-mub
        return self._circ

    @property
    def wash(self):
        if self._wash is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._wash = zeros(num)
            for i, strp in enumerate(sys.strps):
                if not strp.nohsv:
                    pnla = strp.pnls[0]
                    pnlb = strp.pnls[-1]
                    pinda = pnla.ind
                    pindb = pnlb.ind
                    hindsa = sys.phind[pinda]
                    hindsb = sys.phind[pindb]
                    cnt = 0
                    for hind in hindsa:
                        cnt += 1
                        self._wash[i] -= self.ffwsh[hind]
                    for hind in hindsb:
                        cnt += 1
                        self._wash[i] += self.ffwsh[hind]
                    self._wash[i] = self._wash[i]/cnt
        return self._wash

    @property
    def drag(self):
        if self._drag is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._drag = zeros(num)
            for i, strp in enumerate(sys.strps):
                if not strp.noload:
                    self._drag[i] = -self.res.rho*self.wash[i]*self.circ[i]*strp.width/2
        return self._drag

    @property
    def side(self):
        if self._side is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._side = zeros(num)
            for i, strp in enumerate(sys.strps):
                if not strp.noload:
                    if not strp.nohsv:
                        pnla = strp.pnls[0]
                        pnlb = strp.pnls[-1]
                        pinda = pnla.ind
                        pindb = pnlb.ind
                        hindsa = sys.phind[pinda]
                        hindsb = sys.phind[pindb]
                        for hind in hindsa:
                            self._side[i] += self.fffrc[hind].y
                        for hind in hindsb:
                            self._side[i] += self.fffrc[hind].y
        return self._side

    @property
    def lift(self):
        if self._lift is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._lift = zeros(num)
            for i, strp in enumerate(sys.strps):
                if not strp.noload:
                    if not strp.nohsv:
                        pnla = strp.pnls[0]
                        pnlb = strp.pnls[-1]
                        pinda = pnla.ind
                        pindb = pnlb.ind
                        hindsa = sys.phind[pinda]
                        hindsb = sys.phind[pindb]
                        for hind in hindsa:
                            self._lift[i] += self.fffrc[hind].z
                        for hind in hindsb:
                            self._lift[i] += self.fffrc[hind].z
        return self._lift

    @property
    def CDi(self):
        if self._CDi is None:
            Di = self.drag.sum()
            self._CDi = Di/self.res.qfs/self.res.sys.sref
            self._CDi = fix_zero(self._CDi)
        return self._CDi

    @property
    def CY(self):
        if self._CY is None:
            Y = self.side.sum()
            self._CY = Y/self.res.qfs/self.res.sys.sref
            self._CY = fix_zero(self._CY)
        return self._CY

    @property
    def CL(self):
        if self._CL is None:
            L = self.lift.sum()
            self._CL = L/self.res.qfs/self.res.sys.sref
            self._CL = fix_zero(self._CL)
        return self._CL

    @property
    def Cl(self):
        if self._Cl is None:
            l = -self.ffmomtot.x
            self._Cl = l/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cl = fix_zero(self._Cl)
        return self._Cl

    @property
    def Cm(self):
        if self._Cm is None:
            m = self.ffmomtot.y
            self._Cm = m/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cm = fix_zero(self._Cm)
        return self._Cm

    @property
    def Cn(self):
        if self._Cn is None:
            n = -self.ffmomtot.z
            self._Cn = n/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cn = fix_zero(self._Cn)
        return self._Cn

    @property
    def e(self):
        if self._e is None:
            if self.CDi == 0.0:
                if self.CL == 0.0 and self.CY == 0.0:
                    self._e = 0.0
                else:
                    self._e = float('nan')
            else:
                self._e = (self.CL**2+self.CY**2)/pi/self.res.sys.ar/self.CDi
                self._e = fix_zero(self._e)
        return self._e


class StabilityResult():
    res: PanelResult = None
    _u: StabilityNearFieldResult = None
    _v: StabilityNearFieldResult = None
    _w: StabilityNearFieldResult = None
    _p: StabilityNearFieldResult = None
    _q: StabilityNearFieldResult = None
    _r: StabilityNearFieldResult = None
    _alpha: StabilityNearFieldResult = None
    _beta: StabilityNearFieldResult = None
    _pbo2v: StabilityNearFieldResult = None
    _qco2v: StabilityNearFieldResult = None
    _rbo2v: StabilityNearFieldResult = None
    _pdbo2V: StabilityNearFieldResult = None
    _qdco2V: StabilityNearFieldResult = None
    _rdbo2V: StabilityNearFieldResult = None
    _xnp: float = None
    _sprat: float = None

    def __init__(self, res: PanelResult) -> None:
        self.res = res

    @property
    def u(self) -> StabilityNearFieldResult:
        if self._u is None:
            dvfs = Vector(1.0, 0.0, 0.0)
            self._u = StabilityNearFieldResult(self.res, dvfs = dvfs)
        return self._u

    @property
    def v(self) -> StabilityNearFieldResult:
        if self._v is None:
            dvfs = Vector(0.0, 1.0, 0.0)
            self._v = StabilityNearFieldResult(self.res, dvfs = dvfs)
        return self._v

    @property
    def w(self) -> StabilityNearFieldResult:
        if self._w is None:
            dvfs = Vector(0.0, 0.0, 1.0)
            self._w = StabilityNearFieldResult(self.res, dvfs = dvfs)
        return self._w

    @property
    def p(self) -> StabilityNearFieldResult:
        if self._p is None:
            dpqr = Vector(1.0, 0.0, 0.0)
            dofs = self.res.scs.vector_to_global(dpqr)
            self._p = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._p

    @property
    def q(self) -> StabilityNearFieldResult:
        if self._q is None:
            dpqr = Vector(0.0, 1.0, 0.0)
            dofs = self.res.scs.vector_to_global(dpqr)
            self._q = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._q

    @property
    def r(self) -> StabilityNearFieldResult:
        if self._r is None:
            dpqr = Vector(0.0, 0.0, 1.0)
            dofs = self.res.scs.vector_to_global(dpqr)
            self._r = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._r

    @property
    def alpha(self) -> StabilityNearFieldResult:
        if self._alpha is None:
            dvfs = self.res.dvfsa
            dofs = Vector(
                self.res.dscsa['x'].dot(self.res.pqr),
                self.res.dscsa['y'].dot(self.res.pqr),
                self.res.dscsa['z'].dot(self.res.pqr)
            )
            self._alpha = StabilityNearFieldResult(self.res, dvfs = dvfs, dofs = dofs,
                                                   dacs = self.res.dacsa,
                                                   dscs = self.res.dscsa)
        return self._alpha

    @property
    def beta(self) -> StabilityNearFieldResult:
        if self._beta is None:
            dvfs = self.res.dvfsb
            self._beta = StabilityNearFieldResult(self.res, dvfs = dvfs,
                                                  dacs = self.res.dacsb,
                                                  dscs = self.res.dscsb)
        return self._beta

    @property
    def pbo2v(self) -> StabilityNearFieldResult:
        if self._pbo2v is None:
            dpqr = Vector(2*self.res.speed/self.res.sys.bref, 0.0, 0.0)
            dofs = self.res.scs.vector_to_global(dpqr)
            self._pbo2v = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._pbo2v

    @property
    def qco2v(self) -> StabilityNearFieldResult:
        if self._qco2v is None:
            dpqr = Vector(0.0, 2*self.res.speed/self.res.sys.cref, 0.0)
            dofs = self.res.scs.vector_to_global(dpqr)
            self._qco2v = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._qco2v

    @property
    def rbo2v(self) -> StabilityNearFieldResult:
        if self._rbo2v is None:
            dpqr = Vector(0.0, 0.0, 2*self.res.speed/self.res.sys.bref)
            dofs = self.res.scs.vector_to_global(dpqr)
            self._rbo2v = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._rbo2v

    @property
    def pdbo2V(self) -> StabilityNearFieldResult:
        if self._pdbo2V is None:
            dofs = Vector(2*self.res.speed/self.res.sys.bref, 0.0, 0.0)
            self._pdbo2V = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._pdbo2V

    @property
    def qdco2V(self) -> StabilityNearFieldResult:
        if self._qdco2V is None:
            dofs = Vector(0.0, 2*self.res.speed/self.res.sys.cref, 0.0)
            self._qdco2V = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._qdco2V

    @property
    def rdbo2V(self) -> StabilityNearFieldResult:
        if self._rdbo2V is None:
            dofs = Vector(0.0, 0.0, 2*self.res.speed/self.res.sys.bref)
            self._rdbo2V = StabilityNearFieldResult(self.res, dofs = dofs)
        return self._rdbo2V

    @property
    def xnp(self) -> float:
        if self._xnp is None:
            xcg = self.res.rcg.x
            CLa = self.alpha.CL
            CMa = self.alpha.Cm
            c = self.res.sys.cref
            self._xnp = xcg - c*CMa/CLa
        return self._xnp

    @property
    def sprat(self) -> float:
        if self._sprat is None:
            Clb = self.beta.Cl
            Cnb = self.beta.Cn
            Cnr = self.rbo2v.Cn
            Clr = self.rbo2v.Cl
            if Clb == 0.0 and Clr == 0.0:
                self._sprat = float('nan')
            elif Cnb == 0.0 and Cnr == 0.0:
                self._sprat = float('nan')
            else:
                self._sprat = Clb*Cnr/(Clr*Cnb)
        return self._sprat

    def system_aerodynamic_matrix(self):
        A = zeros((6, 6))
        F = self.u.dfrctot
        A[0, 0], A[1, 0], A[2, 0] = F.x, F.y, F.z
        M = self.res.scs.vector_to_local(self.u.dmomtot)
        A[3, 0], A[4, 0], A[5, 0] = M.x, M.y, M.z
        F = self.v.dfrctot
        A[0, 1], A[1, 1], A[2, 1] = F.x, F.y, F.z
        M = self.res.scs.vector_to_local(self.v.dmomtot)
        A[3, 1], A[4, 1], A[5, 1] = M.x, M.y, M.z
        F = self.w.dfrctot
        A[0, 2], A[1, 2], A[2, 2] = F.x, F.y, F.z
        M = self.res.scs.vector_to_local(self.w.dmomtot)
        A[3, 2], A[4, 2], A[5, 2] = M.x, M.y, M.z
        F = self.p.dfrctot
        A[0, 3], A[1, 3], A[2, 3] = F.x, F.y, F.z
        M = self.res.scs.vector_to_local(self.p.dmomtot)
        A[3, 3], A[4, 3], A[5, 3] = M.x, M.y, M.z
        F = self.q.dfrctot
        A[0, 4], A[1, 4], A[2, 4] = F.x, F.y, F.z
        M = self.res.scs.vector_to_local(self.q.dmomtot)
        A[3, 4], A[4, 4], A[5, 4] = M.x, M.y, M.z
        F = self.r.dfrctot
        A[0, 5], A[1, 5], A[2, 5] = F.x, F.y, F.z
        M = self.res.scs.vector_to_local(self.r.dmomtot)
        A[3, 5], A[4, 5], A[5, 5] = M.x, M.y, M.z
        return A

    @property
    def stability_derivatives(self):

        from . import sfrm
        report = MDReport()
        report.add_heading('Stability Derivatives', 2)
        table = report.add_table()
        table.add_column('CLa', sfrm, data=[self.alpha.CL])
        table.add_column('CYa', sfrm, data=[self.alpha.CY])
        table.add_column('Cla', sfrm, data=[self.alpha.Cl])
        table.add_column('Cma', sfrm, data=[self.alpha.Cm])
        table.add_column('Cna', sfrm, data=[self.alpha.Cn])
        table = report.add_table()
        table.add_column('CLb', sfrm, data=[self.beta.CL])
        table.add_column('CYb', sfrm, data=[self.beta.CY])
        table.add_column('Clb', sfrm, data=[self.beta.Cl])
        table.add_column('Cmb', sfrm, data=[self.beta.Cm])
        table.add_column('Cnb', sfrm, data=[self.beta.Cn])
        table = report.add_table()
        table.add_column('CLp', sfrm, data=[self.pbo2v.CL])
        table.add_column('CYp', sfrm, data=[self.pbo2v.CY])
        table.add_column('Clp', sfrm, data=[self.pbo2v.Cl])
        table.add_column('Cmp', sfrm, data=[self.pbo2v.Cm])
        table.add_column('Cnp', sfrm, data=[self.pbo2v.Cn])
        table = report.add_table()
        table.add_column('CLq', sfrm, data=[self.qco2v.CL])
        table.add_column('CYq', sfrm, data=[self.qco2v.CY])
        table.add_column('Clq', sfrm, data=[self.qco2v.Cl])
        table.add_column('Cmq', sfrm, data=[self.qco2v.Cm])
        table.add_column('Cnq', sfrm, data=[self.qco2v.Cn])
        table = report.add_table()
        table.add_column('CLr', sfrm, data=[self.rbo2v.CL])
        table.add_column('CYr', sfrm, data=[self.rbo2v.CY])
        table.add_column('Clr', sfrm, data=[self.rbo2v.Cl])
        table.add_column('Cmr', sfrm, data=[self.rbo2v.Cm])
        table.add_column('Cnr', sfrm, data=[self.rbo2v.Cn])
        report.add_heading(f'Neutral Point Xnp = {self.xnp:.6f}', 3)
        report.add_heading(f'Clb.Cnr/(Clr.Cnb) = {self.sprat:.6f} (> 1 if spirally stable)', 3)
        return report

    @property
    def stability_derivatives_body(self) -> MDReport:
        from . import sfrm
        report = MDReport()
        report.add_heading('Stability Derivatives Body Axis', 2)
        table = report.add_table()
        table.add_column('Cxu', sfrm, data=[self.u.Cx])
        table.add_column('Cyu', sfrm, data=[self.u.Cy])
        table.add_column('Czu', sfrm, data=[self.u.Cz])
        table.add_column('Clu', sfrm, data=[self.u.Cmx])
        table.add_column('Cmu', sfrm, data=[self.u.Cmy])
        table.add_column('Cnu', sfrm, data=[self.u.Cmz])
        table = report.add_table()
        table.add_column('Cxv', sfrm, data=[self.v.Cx])
        table.add_column('Cyv', sfrm, data=[self.v.Cy])
        table.add_column('Czv', sfrm, data=[self.v.Cz])
        table.add_column('Clv', sfrm, data=[self.v.Cmx])
        table.add_column('Cmv', sfrm, data=[self.v.Cmy])
        table.add_column('Cnv', sfrm, data=[self.v.Cmz])
        table = report.add_table()
        table.add_column('Cxw', sfrm, data=[self.w.Cx])
        table.add_column('Cyw', sfrm, data=[self.w.Cy])
        table.add_column('Czw', sfrm, data=[self.w.Cz])
        table.add_column('Clw', sfrm, data=[self.w.Cmx])
        table.add_column('Cmw', sfrm, data=[self.w.Cmy])
        table.add_column('Cnw', sfrm, data=[self.w.Cmz])
        table = report.add_table()
        table.add_column('Cxp', sfrm, data=[self.pdbo2V.Cx])
        table.add_column('Cyp', sfrm, data=[self.pdbo2V.Cy])
        table.add_column('Czp', sfrm, data=[self.pdbo2V.Cz])
        table.add_column('Clp', sfrm, data=[self.pdbo2V.Cmx])
        table.add_column('Cmp', sfrm, data=[self.pdbo2V.Cmy])
        table.add_column('Cnp', sfrm, data=[self.pdbo2V.Cmz])
        table = report.add_table()
        table.add_column('Cxq', sfrm, data=[self.qdco2V.Cx])
        table.add_column('Cyq', sfrm, data=[self.qdco2V.Cy])
        table.add_column('Czq', sfrm, data=[self.qdco2V.Cz])
        table.add_column('Clq', sfrm, data=[self.qdco2V.Cmx])
        table.add_column('Cmq', sfrm, data=[self.qdco2V.Cmy])
        table.add_column('Cnq', sfrm, data=[self.qdco2V.Cmz])
        table = report.add_table()
        table.add_column('Cxr', sfrm, data=[self.rdbo2V.Cx])
        table.add_column('Cyr', sfrm, data=[self.rdbo2V.Cy])
        table.add_column('Czr', sfrm, data=[self.rdbo2V.Cz])
        table.add_column('Clr', sfrm, data=[self.rdbo2V.Cmx])
        table.add_column('Cmr', sfrm, data=[self.rdbo2V.Cmy])
        table.add_column('Cnr', sfrm, data=[self.rdbo2V.Cmz])
        report.add_object(table)
        return report

    def __str__(self) -> str:
        return self.stability_derivatives._repr_markdown_()

    def _repr_markdown_(self) -> str:
        return self.__str__()


def fix_zero(value: float, tol: float=1e-8) -> float:
    if abs(value) < tol:
        value = 0.0
    return value
