from typing import TYPE_CHECKING, Any

from matplotlib.pyplot import figure
from numpy import asarray, concatenate, cos, degrees, pi, radians, sin, zeros
from numpy.linalg import eig, norm, solve
from py2md.classes import MDReport
from ...tools.mass import MassObject
from pygeom.geom3d import Coordinate, Vector

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from .constantcontrol import ControlObject
    from .constantgeometry import ConstantSurface
    from .constantsystem import ConstantSystem

SET_SET = {'_speed', '_alpha', '_beta',
           '_pbo2v', '_qco2v', '_rbo2v',
           '_ctrls', '_rcg', '_rho',
           '_mach', '_trim_result'}


GRAVACC = 9.80665 # m/s^2


class TransformDerivative:
    dirx: Vector
    diry: Vector
    dirz: Vector

    __slots__ = ('dirx', 'diry', 'dirz')

    def __init__(self, dirx: Vector, diry: Vector, dirz: Vector) -> None:
        self.dirx = dirx
        self.diry = diry
        self.dirz = dirz

    def vector_to_global(self, vec: Vector) -> Vector:
        """Transforms a vector from this local coordinate to global."""
        dirx = Vector(self.dirx.x, self.diry.x, self.dirz.x)
        diry = Vector(self.dirx.y, self.diry.y, self.dirz.y)
        dirz = Vector(self.dirx.z, self.diry.z, self.dirz.z)
        x = vec.dot(dirx)
        y = vec.dot(diry)
        z = vec.dot(dirz)
        return vec.__class__(x, y, z)


class ConstantResult:
    name: str
    system: 'ConstantSystem'
    rho: float
    mach: float
    speed: float
    alpha: float
    beta: float
    pbo2v: float
    qco2v: float
    rbo2v: float
    ctrls: dict[str, float]
    rcg: Vector
    CDo: float
    mass: MassObject
    trim_result: 'TrimResult'
    _acs: Coordinate
    _scs: Coordinate
    _dacsa: TransformDerivative
    _dscsa: TransformDerivative
    _qfs: float
    _vfs: Vector
    _dvfsa: Vector
    _dvfsb: Vector
    _pqr: Vector
    _ofs: Vector
    _nrel: Vector
    _nnormal_approx: Vector
    _nnrml: Vector
    _nvel: Vector
    _drel: Vector
    _dnormal_approx: Vector
    _dnrml: Vector
    _dvel: Vector
    _grel: Vector
    _sig: 'NDArray'
    _mud: 'NDArray'
    _mun: 'NDArray'
    _muw: 'NDArray'
    _result: 'DirectResult'
    _mup: 'NDArray'
    _ctrl_results: dict[str, list['DirectResult']]
    _ctrl_der_res: dict[str, 'StabilityDirectResult']
    _stab_der_res: 'StabilityResult'
    _modes: 'StabilityModes'
    _strip_results: dict[str, 'StripResult']

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str, system: 'ConstantSystem') -> None:
        self.name = name
        self.system = system
        self.rho = 1.0
        self.mach = 0.0
        self.speed = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        self.pbo2v = 0.0
        self.qco2v = 0.0
        self.rbo2v = 0.0
        self.mass = MassObject()
        self.ctrls = {control: 0.0 for control in self.system.ctrls}
        self.rcg = self.system.rref.copy()
        self.CDo = self.system.CDo
        self.trim_result = TrimResult(self)
        self.reset()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def set_state(self, **kwargs: dict[str, float]) -> None:
        self.speed = kwargs.get('speed', self.speed)
        self.alpha = kwargs.get('alpha', self.alpha)
        self.beta = kwargs.get('beta', self.beta)
        self.pbo2v = kwargs.get('pbo2v', self.pbo2v)
        self.qco2v = kwargs.get('qco2v', self.qco2v)
        self.rbo2v = kwargs.get('rbo2v', self.rbo2v)
        self.reset()

    def get_state(self) -> dict[str, float]:
        return {'speed': self.speed, 'alpha': self.alpha, 'beta': self.beta,
                'pbo2v': self.pbo2v, 'qco2v': self.qco2v, 'rbo2v': self.rbo2v}

    def set_controls(self, **kwargs: dict[str, float]) -> None:
        for control in self.ctrls:
            self.ctrls[control] = kwargs.get(control, self.ctrls[control])
        self.reset()

    def get_controls(self) -> dict[str, float]:
        return {control: self.ctrls[control] for control in self.ctrls}

    def calc_coordinate_systems(self) -> None:
        pnt = self.rcg
        cosal, sinal = trig_angle(self.alpha)
        cosbt, sinbt = trig_angle(self.beta)
        self._vfs = Vector(cosal*cosbt, -sinbt, sinal*cosbt)*self.speed
        self._dvfsa = Vector(-sinal*cosbt, 0.0, cosal*cosbt)*self.speed
        self._dvfsb = Vector(-cosal*sinbt, -cosbt, -sinal*sinbt)*self.speed
        # Aerodynamic Coordinate System
        acs_dirx = Vector(cosal, 0.0, sinal)
        acs_diry = Vector(0.0, 1.0, 0.0)
        self._acs = Coordinate(pnt, acs_dirx, acs_diry)
        # Stability Coordinate System
        scs_dirx = Vector(-cosal, 0.0, -sinal)
        scs_diry = Vector(0.0, 1.0, 0.0)
        self._scs = Coordinate(pnt, scs_dirx, scs_diry)
        # Derivative of Aerodynamic Coordinate System wrt alpha
        dacsa_dirx = Vector(-sinal, 0.0, cosal)
        dacsa_diry = Vector.zeros()
        dacsa_dirz = Vector(-cosal, 0.0, -sinal)
        self._dacsa = TransformDerivative(dacsa_dirx, dacsa_diry, dacsa_dirz)
        # Derivative of Stability Coordinate System wrt alpha
        dscsa_dirx = Vector(sinal, 0.0, -cosal)
        dscsa_diry = Vector.zeros()
        dscsa_dirz = Vector(cosal, 0.0, sinal)
        self._dscsa = TransformDerivative(dscsa_dirx, dscsa_diry, dscsa_dirz)

    @property
    def acs(self) -> 'Coordinate':
        if self._acs is None:
            self.calc_coordinate_systems()
        return self._acs

    @property
    def scs(self) -> 'Coordinate':
        if self._scs is None:
            self.calc_coordinate_systems()
        return self._scs

    @property
    def dacsa(self) -> TransformDerivative:
        if self._dacsa is None:
            self.calc_coordinate_systems()
        return self._dacsa

    @property
    def dscsa(self) -> TransformDerivative:
        if self._dscsa is None:
            self.calc_coordinate_systems()
        return self._dscsa

    @property
    def qfs(self) -> float:
        if self._qfs is None:
            self._qfs = self.rho*self.speed**2/2
        return self._qfs

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
            p = self.pbo2v*self.speed*2/self.system.bref
            q = self.qco2v*self.speed*2/self.system.cref
            r = self.rbo2v*self.speed*2/self.system.bref
            self._pqr = Vector(p, q, r)
        return self._pqr

    @property
    def ofs(self) -> Vector:
        if self._ofs is None:
            self._ofs = self.scs.vector_to_global(self.pqr)
        return self._ofs

    @property
    def nrel(self) -> Vector:
        if self._nrel is None:
            self._nrel = self.system.npoints - self.rcg
        return self._nrel

    @property
    def nvel(self) -> Vector:
        if self._nvel is None:
            self._nvel = self.vfs + self.nrel.cross(self.ofs)
        return self._nvel

    @property
    def nnormal_approx(self) -> Vector:
        if self._nnormal_approx is None:
            slc = slice(self.system.num_dpanels, None)
            self._nnormal_approx = self.system.nnormal.copy()
            for control, ctrl in self.system.ctrls.items():
                value = radians(self.ctrls[control])
                if value >= 0.0:
                    pos_nrml = ctrl.normal_change_approx[slc, 0]
                    self._nnormal_approx += pos_nrml*value
                else:
                    neg_nrml = ctrl.normal_change_approx[slc, 1]
                    self._nnormal_approx += neg_nrml*value
        return self._nnormal_approx

    @property
    def nnrml(self) -> Vector:
        if self._nnrml is None:
            self._nnrml = Vector.zeros((self.system.num_npanels, 1 + self.system.num_ctrls))
            self._nnrml[:, 0] = self.system.nnormal
            for i, ctrl in enumerate(self.system.ctrls.values(), start=1):
                value = radians(self.ctrls[ctrl.name])
                nchg, nder = ctrl.normal_change_and_derivative(value)
                nnormal_change = nchg[self.system.num_dpanels:]
                self._nnrml[:, 0] += nnormal_change
                self._nnrml[:, i] = nder[self.system.num_dpanels:]
        return self._nnrml

    @property
    def drel(self) -> Vector:
        if self._drel is None:
            self._drel = self.system.dpoints - self.rcg
        return self._drel

    @property
    def dnormal_approx(self) -> Vector:
        if self._dnormal_approx is None:
            slc = slice(0, self.system.num_dpanels)
            self._dnormal_approx = self.system.dnormal.copy()
            for control, ctrl in self.system.ctrls.items():
                value = radians(self.ctrls[control])
                if value >= 0.0:
                    pos_nrml = ctrl.normal_change_approx[slc, 0]
                    self._dnormal_approx += pos_nrml*value
                else:
                    neg_nrml = ctrl.normal_change_approx[slc, 1]
                    self._dnormal_approx += neg_nrml*value
        return self._dnormal_approx

    @property
    def dnrml(self) -> Vector:
        if self._dnrml is None:
            self._dnrml = Vector.zeros((self.system.num_dpanels, 1 + self.system.num_ctrls))
            self._dnrml[:, 0] = self.system.dnormal
            for i, ctrl in enumerate(self.system.ctrls.values(), start=1):
                value = radians(self.ctrls[ctrl.name])
                dchg, dder = ctrl.normal_change_and_derivative(value)
                dnormal_change = dchg[:self.system.num_dpanels]
                self._dnrml[:, 0] += dnormal_change
                self._dnrml[:, i] = dder[:self.system.num_dpanels]
        return self._dnrml

    @property
    def dvel(self) -> Vector:
        if self._dvel is None:
            self._dvel = self.vfs + self.drel.cross(self.ofs)
        return self._dvel

    def calculate_sig(self, vfs: Vector | None = None,
                      ofs: Vector | None = None) -> 'NDArray':
        sig = zeros(self.system.num_dpanels)
        if vfs is not None:
            sig += self.system.unsig[:, 0].dot(vfs)
        if ofs is not None:
            sig += self.system.unsig[:, 1].dot(ofs)
        for ctrl in self.ctrls:
            ind1, ind2, ind3, ind4 = self.system.ctrls[ctrl].index
            value = radians(self.ctrls[ctrl])
            if value >= 0.0:
                if vfs is not None:
                    sig += self.system.unsig[:, ind1].dot(vfs)*value
                if ofs is not None:
                    sig += self.system.unsig[:, ind2].dot(ofs)*value
            else:
                if vfs is not None:
                    sig += self.system.unsig[:, ind3].dot(vfs)*value
                if ofs is not None:
                    sig += self.system.unsig[:, ind4].dot(ofs)*value
        return sig

    @property
    def sig(self) -> 'NDArray':
        if self._sig is None:
            self._sig = self.calculate_sig(self.vfs, self.ofs)
        return self._sig

    def calculate_mud(self, vfs: Vector | None = None,
                      ofs: Vector | None = None) -> 'NDArray':
        mud = zeros(self.system.num_dpanels)
        if vfs is not None:
            mud += self.system.unmud[:, 0].dot(vfs)
        if ofs is not None:
            mud += self.system.unmud[:, 1].dot(ofs)
        for ctrl in self.ctrls:
            ind1, ind2, ind3, ind4 = self.system.ctrls[ctrl].index
            value = radians(self.ctrls[ctrl])
            if value >= 0.0:
                if vfs is not None:
                    mud += self.system.unmud[:, ind1].dot(vfs)*value
                if ofs is not None:
                    mud += self.system.unmud[:, ind2].dot(ofs)*value
            else:
                if vfs is not None:
                    mud += self.system.unmud[:, ind3].dot(vfs)*value
                if ofs is not None:
                    mud += self.system.unmud[:, ind4].dot(ofs)*value
        return mud

    @property
    def mud(self) -> 'NDArray':
        if self._mud is None:
            self._mud = self.calculate_mud(self.vfs, self.ofs)
        return self._mud

    def calculate_mun(self, vfs: Vector | None = None,
                      ofs: Vector | None = None) -> 'NDArray':
        mun = zeros(self.system.num_npanels)
        if vfs is not None:
            mun += self.system.unmun[:, 0].dot(vfs)
        if ofs is not None:
            mun += self.system.unmun[:, 1].dot(ofs)
        for ctrl in self.ctrls:
            ind1, ind2, ind3, ind4 = self.system.ctrls[ctrl].index
            value = radians(self.ctrls[ctrl])
            if value >= 0.0:
                if vfs is not None:
                    mun += self.system.unmun[:, ind1].dot(vfs)*value
                if ofs is not None:
                    mun += self.system.unmun[:, ind2].dot(ofs)*value
            else:
                if vfs is not None:
                    mun += self.system.unmun[:, ind3].dot(vfs)*value
                if ofs is not None:
                    mun += self.system.unmun[:, ind4].dot(ofs)*value
        return mun

    @property
    def mun(self) -> 'NDArray':
        if self._mun is None:
            self._mun = self.calculate_mun(self.vfs, self.ofs)
        return self._mun

    def calculate_muw(self, vfs: Vector | None = None,
                      ofs: Vector | None = None) -> 'NDArray':
        muw = zeros(self.system.num_wpanels)
        if vfs is not None:
            muw += self.system.unmuw[:, 0].dot(vfs)
        if ofs is not None:
            muw += self.system.unmuw[:, 1].dot(ofs)
        for ctrl in self.ctrls:
            ind1, ind2, ind3, ind4 = self.system.ctrls[ctrl].index
            value = radians(self.ctrls[ctrl])
            if value >= 0.0:
                if vfs is not None:
                    muw += self.system.unmuw[:, ind1].dot(vfs)*value
                if ofs is not None:
                    muw += self.system.unmuw[:, ind2].dot(ofs)*value
            else:
                if vfs is not None:
                    muw += self.system.unmuw[:, ind3].dot(vfs)*value
                if ofs is not None:
                    muw += self.system.unmuw[:, ind4].dot(ofs)*value
        return muw

    @property
    def muw(self) -> 'NDArray':
        if self._muw is None:
            self._muw = self.calculate_muw(self.vfs, self.ofs)
        return self._muw

    @property
    def grel(self) -> Vector:
        if self._grel is None:
            self._grel = self.system.gridvec - self.rcg
        return self._grel

    @property
    def result(self) -> 'DirectResult':
        if self._result is None:
            self._result = DirectResult(self, self.sig, self.mud, self.mun,
                                        self.muw, vfs = self.vfs,
                                        ofs = self.ofs)
        return self._result

    @property
    def mup(self) -> 'NDArray':
        if self._mup is None:
            self._mup = concatenate((self.mud, self.mun))
        return self._mup

    @property
    def ctrl_results(self) -> dict[str, list['ControlResult']]:
        if self._ctrl_results is None:
            self._ctrl_results = {}
            for ctrl in self.system.ctrls.values():
                self._ctrl_results[ctrl.name] = []
                for ctrlobj in ctrl.control_objects:
                    muc = zeros(self.system.num_panels)
                    muc[ctrlobj.panel_index] = self.mup[ctrlobj.panel_index]
                    mud = muc[:self.system.num_dpanels]
                    mun = muc[self.system.num_dpanels:]
                    muw = self.system.cmat@muc
                    ctrl_result = ControlResult(self, ctrlobj, mud, mun, muw)
                    self._ctrl_results[ctrl.name].append(ctrl_result)
        return self._ctrl_results

    @property
    def stab_der_res(self) -> 'StabilityResult':
        if self._stab_der_res is None:
            self._stab_der_res = StabilityResult(self)
        return self._stab_der_res

    @property
    def ctrl_der_res(self) -> dict[str, 'DirectResult']:
        if self._ctrl_der_res is None:
            self._ctrl_der_res = {}
            for ctrlkey, ctrlval in self.ctrls.items():
                ind1, ind2, ind3, ind4 = self.system.ctrls[ctrlkey].index
                if ctrlval > 0.0:
                    dsig = self.system.unsig[:, ind1].dot(self.vfs)
                    dsig += self.system.unsig[:, ind2].dot(self.ofs)
                    dmud = self.system.unmud[:, ind1].dot(self.vfs)
                    dmud += self.system.unmud[:, ind2].dot(self.ofs)
                    dmun = self.system.unmun[:, ind1].dot(self.vfs)
                    dmun += self.system.unmun[:, ind2].dot(self.ofs)
                    dmuw = self.system.unmuw[:, ind1].dot(self.vfs)
                    dmuw += self.system.unmuw[:, ind2].dot(self.ofs)
                elif ctrlval < 0.0:
                    dsig = self.system.unsig[:, ind3].dot(self.vfs)
                    dsig += self.system.unsig[:, ind4].dot(self.ofs)
                    dmud = self.system.unmud[:, ind3].dot(self.vfs)
                    dmud += self.system.unmud[:, ind4].dot(self.ofs)
                    dmun = self.system.unmun[:, ind3].dot(self.vfs)
                    dmun += self.system.unmun[:, ind4].dot(self.ofs)
                    dmuw = self.system.unmuw[:, ind3].dot(self.vfs)
                    dmuw += self.system.unmuw[:, ind4].dot(self.ofs)
                else:
                    dsig = self.system.unsig[:, ind1].dot(self.vfs)
                    dsig += self.system.unsig[:, ind2].dot(self.ofs)
                    dsig += self.system.unsig[:, ind3].dot(self.vfs)
                    dsig += self.system.unsig[:, ind4].dot(self.ofs)
                    dmud = self.system.unmud[:, ind1].dot(self.vfs)
                    dmud += self.system.unmud[:, ind2].dot(self.ofs)
                    dmud += self.system.unmud[:, ind3].dot(self.vfs)
                    dmud += self.system.unmud[:, ind4].dot(self.ofs)
                    dmun = self.system.unmun[:, ind1].dot(self.vfs)
                    dmun += self.system.unmun[:, ind2].dot(self.ofs)
                    dmun += self.system.unmun[:, ind3].dot(self.vfs)
                    dmun += self.system.unmun[:, ind4].dot(self.ofs)
                    dmuw = self.system.unmuw[:, ind1].dot(self.vfs)
                    dmuw += self.system.unmuw[:, ind2].dot(self.ofs)
                    dmuw += self.system.unmuw[:, ind3].dot(self.vfs)
                    dmuw += self.system.unmuw[:, ind4].dot(self.ofs)
                    dsig = dsig/2.0
                    dmud = dmud/2.0
                    dmun = dmun/2.0
                    dmuw = dmuw/2.0
                ctrl_der_res = StabilityDirectResult(self, dsig, dmud,
                                                     dmun, dmuw)
                self._ctrl_der_res[ctrlkey] = ctrl_der_res
        return self._ctrl_der_res

    @property
    def modes(self) -> 'StabilityModes':
        if self._modes is None:
            self._modes = StabilityModes(self)
        return self._modes

    @property
    def strip_results(self) -> dict[str, 'StripResult']:
        if self._strip_results is None:
            if self.system.geometry is not None:
                self._strip_results = {}
                for surface in self.system.geometry.surfaces.values():
                    self._strip_results[surface.name] = StripResult(self, surface)
        return self._strip_results

    @property
    def stability_derivatives(self) -> MDReport:
        return self.stab_der_res.stability_derivatives

    @property
    def stability_derivatives_body(self) -> MDReport:
        return self.stab_der_res.stability_derivatives_body

    @property
    def control_derivatives(self) -> MDReport:
        from . import sfrm
        report = MDReport()
        report.add_heading('Control Derivatives', 1)
        for i, (control, ctres) in enumerate(self.ctrl_der_res.items(), start=1):
            report.add_heading(f'{control.capitalize()} Derivatives', 2)
            table = report.add_table()
            table.add_column(f'C<sub>L,d{i:d}</sub>', sfrm)
            table.add_column(f'C<sub>Y,d{i:d}</sub>', sfrm)
            table.add_column(f'C<sub>l,d{i:d}</sub>', sfrm)
            table.add_column(f'C<sub>m,d{i:d}</sub>', sfrm)
            table.add_column(f'C<sub>n,d{i:d}</sub>', sfrm)
            table.add_row([ctres.CL, ctres.CY, ctres.Cl, ctres.Cm, ctres.Cn])
        return report

    @property
    def surface_loads(self) -> MDReport:

        from . import get_unit_string

        lstr = get_unit_string('length')
        fstr = get_unit_string('force')
        mstr = get_unit_string('moment')
        Astr = get_unit_string('area')
        dstr = get_unit_string('density')
        vstr = get_unit_string('velocity')
        pstr = get_unit_string('pressure')

        report = MDReport()
        report.add_heading('Surface Loads', 2)
        table = report.add_table()
        table.add_column(f'x<sub>ref</sub>{lstr}', '.3f', data=[self.rcg.x])
        table.add_column(f'y<sub>ref</sub>{lstr}', '.3f', data=[self.rcg.y])
        table.add_column(f'z<sub>ref</sub>{lstr}', '.3f', data=[self.rcg.z])
        table1 = report.add_table()
        table1.add_column('Name', 's')
        table1.add_column(f'F<sub>x</sub>{fstr}', '.3f')
        table1.add_column(f'F<sub>y</sub>{fstr}', '.3f')
        table1.add_column(f'F<sub>z</sub>{fstr}', '.3f')
        table1.add_column(f'M<sub>x</sub>{mstr}', '.3f')
        table1.add_column(f'M<sub>y</sub>{mstr}', '.3f')
        table1.add_column(f'M<sub>z</sub>{mstr}', '.3f')
        table2 = report.add_table()
        table2.add_column('Name', 's')
        table2.add_column(f'Area{Astr}', '.3f')
        table2.add_column(f'D<sub>i</sub>{fstr}', '.3f')
        table2.add_column(f'Y{fstr}', '.3f')
        table2.add_column(f'L{fstr}', '.3f')
        table2.add_column(f'C<sub>Di</sub>', '.7f')
        table2.add_column(f'C<sub>Y</sub>', '.5f')
        table2.add_column(f'C<sub>L</sub>', '.5f')
        Ditot = 0.0
        Ytot = 0.0
        Ltot = 0.0
        for surface in self.system.geometry.surfaces.values():
            area = surface.area
            index = surface.ngridindex
            force = self.result.ngfrc[index].sum()
            moment = self.result.ngmom[index].sum()
            table1.add_row([surface.name,
                            force.x, force.y, force.z,
                            moment.x, moment.y, moment.z])
            if area > 0.0:
                Di = force.dot(self.acs.dirx)
                Y = force.dot(self.acs.diry)
                L = force.dot(self.acs.dirz)
                CDi = Di/self.qfs/area
                CY = Y/self.qfs/area
                CL = L/self.qfs/area
                table2.add_row([surface.name, area, Di, Y, L, CDi, CY, CL])
                Ditot += Di
                Ytot += Y
                Ltot += L
        frc = self.result.nfrc
        mom = self.result.nmom
        table1.add_row(['Total', frc.x, frc.y, frc.z, mom.x, mom.y, mom.z])
        table = report.add_table()
        table.add_column(f'Density{dstr}', '.3f', data=[self.rho])
        table.add_column(f'Speed{vstr}', '.3f', data=[self.speed])
        table.add_column(f'Dynamic Pressure{pstr}', '.1f', data=[self.qfs])
        table2.add_row(['Total', self.system.sref, Ditot, Ytot, Ltot,
                        self.result.CDi, self.result.CY, self.result.CL])
        return report

    @property
    def control_loads(self) -> MDReport:

        from . import cfrm, get_unit_string

        lstr = get_unit_string('length')
        mstr = get_unit_string('moment')

        report = MDReport()
        report.add_heading('Control Loads', 2)
        table = report.add_table()
        table.add_column('Name', 's')
        table.add_column(f'h<sub>x</sub>{lstr}', '.3f')
        table.add_column(f'h<sub>y</sub>{lstr}', '.3f')
        table.add_column(f'h<sub>z</sub>{lstr}', '.3f')
        table.add_column(f'H<sub>m</sub>{mstr}', '.3f')
        table.add_column(f'C<sub>h</sub>', cfrm)

        for ctrlobj_results in self.ctrl_results.values():
            for ctrlobj_result in ctrlobj_results:
                ctrl_obj = ctrlobj_result.ctrl_obj
                name = ctrlobj_result.ctrl_obj.name
                hpx, hpx, hpy = ctrl_obj.point.to_xyz()
                Hm = ctrlobj_result.hmom
                Ch = ctrlobj_result.Ch

                table.add_row([name.capitalize(), hpx, hpx, hpy, Hm, Ch])

        table = report.add_table()
        table.add_column('Name', 's')
        table.add_column('C<sub>D</sub>', cfrm)
        table.add_column('C<sub>Y</sub>', cfrm)
        table.add_column('C<sub>L</sub>', cfrm)
        table.add_column('C<sub>l</sub>', cfrm)
        table.add_column('C<sub>m</sub>', cfrm)
        table.add_column('C<sub>n</sub>', cfrm)

        for ctrlobj_results in self.ctrl_results.values():
            for ctrlobj_result in ctrlobj_results:
                name = ctrlobj_result.ctrl_obj.name
                CD = ctrlobj_result.CD
                CY = ctrlobj_result.CY
                CL = ctrlobj_result.CL
                Cl = ctrlobj_result.Cl
                Cm = ctrlobj_result.Cm
                Cn = ctrlobj_result.Cn

                table.add_row([name.capitalize(), CD, CY, CL, Cl, Cm, Cn])

        return report

    def to_mdobj(self) -> MDReport:

        from . import cfrm, dfrm, efrm, get_unit_string

        lstr = get_unit_string('length')
        dstr = get_unit_string('density')
        vstr = get_unit_string('velocity')

        report = MDReport()
        report.add_heading(f'Constant Result {self.name:s} for {self.system.name:s}', 1)

        table = report.add_table()
        table.add_column(f'Density{dstr}', cfrm, data=[self.rho])
        table.add_column(f'Speed{vstr}', cfrm, data=[self.speed])
        table.add_column(f'x<sub>cg</sub>{lstr}', '.5f', data=[self.rcg.x])
        table.add_column(f'y<sub>cg</sub>{lstr}', '.5f', data=[self.rcg.y])
        table.add_column(f'z<sub>cg</sub>{lstr}', '.5f', data=[self.rcg.z])

        table = report.add_table()
        table.add_column('Alpha (deg)', cfrm, data=[self.alpha])
        table.add_column('Beta (deg)', cfrm, data=[self.beta])
        table.add_column('pb/2V (rad)', cfrm, data=[self.pbo2v])
        table.add_column('qc/2V (rad)', cfrm, data=[self.qco2v])
        table.add_column('rb/2V (rad)', cfrm, data=[self.rbo2v])

        if len(self.ctrls) > 0:
            table = report.add_table()
            for control in self.ctrls:
                ctrl = self.ctrls[control]
                control = control.capitalize()
                table.add_column(f'{control} (deg)', cfrm, data=[ctrl])

        table = report.add_table()
        table.add_column('C<sub>x</sub>', cfrm, data=[self.result.Cx])
        table.add_column('C<sub>y</sub>', cfrm, data=[self.result.Cy])
        table.add_column('C<sub>z</sub>', cfrm, data=[self.result.Cz])

        table = report.add_table()
        table.add_column('C<sub>D</sub>', dfrm, data=[self.result.CD])
        table.add_column('C<sub>Y</sub>', cfrm, data=[self.result.CY])
        table.add_column('C<sub>L</sub>', cfrm, data=[self.result.CL])
        table.add_column('C<sub>l</sub>', cfrm, data=[self.result.Cl])
        table.add_column('C<sub>m</sub>', cfrm, data=[self.result.Cm])
        table.add_column('C<sub>n</sub>', cfrm, data=[self.result.Cn])

        table = report.add_table()
        table.add_column('C<sub>Do</sub>', dfrm, data=[self.result.CDo])
        table.add_column('C<sub>Di</sub>', dfrm, data=[self.result.CDi])
        table.add_column('C<sub>D</sub>', dfrm, data=[self.result.CD])
        table.add_column('e', efrm, data=[self.result.e])
        table.add_column('L/D', '.1f', data=[self.result.lod])

        return report

    def __str__(self) -> str:
        return self.to_mdobj()._repr_markdown_()

    def _repr_markdown_(self) -> str:
        return self.to_mdobj()._repr_markdown_()

    def __repr__(self) -> str:
        return f'ConstantResult({self.name:s})'


class DirectResult:
    result: ConstantResult
    sig: 'NDArray'
    mud: 'NDArray'
    mun: 'NDArray'
    muw: 'NDArray'
    vfs: Vector | None
    ofs: Vector | None
    _qS: float
    _ngvel: Vector
    _nlvec: Vector
    _ngfrc: Vector
    _ngmom: Vector
    _nfrc: Vector
    _nmom: Vector
    _drago: float
    _dragi: float
    _drag: float
    _side: float
    _lift: float
    _roll: float
    _pitch: float
    _yaw: float
    _CDi: float
    _CD: float
    _CY: float
    _CL: float
    _Cl: float
    _Cm: float
    _Cn: float
    _e: float
    _lod: float
    _Cx: float
    _Cy: float
    _Cz: float

    __slots__ = tuple(__annotations__)

    def __init__(self, result: ConstantResult, sig: 'NDArray',
                 mud: 'NDArray', mun: 'NDArray', muw: 'NDArray',
                 vfs: Vector | None = None, ofs: Vector | None = None) -> None:
        self.result = result
        self.sig = sig
        self.mud = mud
        self.mun = mun
        self.muw = muw
        self.vfs = vfs
        self.ofs = ofs
        self.reset()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def system(self) -> 'ConstantSystem':
        return self.result.system

    @property
    def rho(self) -> float:
        return self.result.rho

    @property
    def qfs(self) -> float:
        return self.result.qfs

    @property
    def acs(self) -> Coordinate:
        return self.result.acs

    @property
    def scs(self) -> Coordinate:
        return self.result.scs

    @property
    def grel(self) -> Vector:
        return self.result.grel

    @property
    def sref(self) -> float:
        return self.system.sref

    @property
    def bref(self) -> float:
        return self.system.bref

    @property
    def cref(self) -> float:
        return self.system.cref

    def calc_ngvel(self, sig: 'NDArray', mud: 'NDArray',
                   mun: 'NDArray', muw: 'NDArray',
                   vfs: Vector | None, ofs: Vector | None) -> Vector:
        ngvel = Vector.zeros(self.result.system.num_grids)
        if vfs is not None:
            ngvel += vfs
        if ofs is not None:
            ngvel += self.grel.cross(ofs)
        ngvel += self.system.avgd@mud
        ngvel += self.system.avgs@sig
        ngvel += self.system.avgn@mun
        ngvel += self.system.avgw@muw
        return ngvel

    @property
    def ngvel(self) -> Vector:
        if self._ngvel is None:
            self._ngvel = self.calc_ngvel(self.sig, self.mud, self.mun,
                                          self.muw, self.vfs, self.ofs)
        return self._ngvel

    def calc_nlvec(self, mud: 'NDArray', mun: 'NDArray', muw: 'NDArray') -> Vector:
        nlvec = Vector.zeros(self.system.num_grids)
        nlvec += self.system.blgd@mud
        nlvec += self.system.blgn@mun
        nlvec += self.system.blgw@muw
        return nlvec

    @property
    def nlvec(self) -> Vector:
        if self._nlvec is None:
            self._nlvec = self.calc_nlvec(self.mud, self.mun, self.muw)
        return self._nlvec

    def calc_ngfrc(self, ngvel: Vector, nlvec: Vector) -> Vector:
        return ngvel.cross(nlvec)*self.rho

    @property
    def ngfrc(self) -> Vector:
        if self._ngfrc is None:
            self._ngfrc = self.calc_ngfrc(self.ngvel, self.nlvec)
        return self._ngfrc

    @property
    def ngmom(self) -> Vector:
        if self._ngmom is None:
            self._ngmom = self.grel.cross(self.ngfrc)
        return self._ngmom

    @property
    def nfrc(self) -> Vector:
        if self._nfrc is None:
            self._nfrc = self.ngfrc.sum()
        return self._nfrc

    @property
    def nmom(self) -> Vector:
        if self._nmom is None:
            self._nmom = self.ngmom.sum()
        return self._nmom

    @property
    def qS(self) -> float:
        if self._qS is None:
            self._qS = self.qfs*self.sref
        return self._qS

    @property
    def CDo(self) -> float:
        return self.result.CDo

    def calc_drago(self) -> float:
        return self.qS*self.CDo

    @property
    def drago(self) -> float:
        if self._drago is None:
            self._drago = self.calc_drago()
        return self._drago

    def calc_dragi(self, nfrc: Vector, acs_dirx: Vector) -> float:
        return nfrc.dot(acs_dirx)

    @property
    def dragi(self) -> float:
        if self._dragi is None:
            self._dragi = self.calc_dragi(self.nfrc, self.acs.dirx)
        return self._dragi

    @property
    def drag(self) -> float:
        if self._drag is None:
            self._drag = self.drago + self.dragi
        return self._drag

    def calc_side(self, nfrc: Vector, acs_diry: Vector) -> float:
        return nfrc.dot(acs_diry)

    @property
    def side(self) -> float:
        if self._side is None:
            self._side = self.calc_side(self.nfrc, self.acs.diry)
        return self._side

    def calc_lift(self, nfrc: Vector, acs_dirz: Vector) -> float:
        return nfrc.dot(acs_dirz)

    @property
    def lift(self) -> float:
        if self._lift is None:
            self._lift = self.calc_lift(self.nfrc, self.acs.dirz)
        return self._lift

    def calc_roll(self, nmom: Vector, scs_dirx: Vector) -> float:
        return nmom.dot(scs_dirx)

    @property
    def roll(self) -> float:
        if self._roll is None:
            self._roll = self.calc_roll(self.nmom, self.scs.dirx)
        return self._roll

    def calc_pitch(self, nmom: Vector, scs_diry: Vector) -> float:
        return nmom.dot(scs_diry)

    @property
    def pitch(self) -> float:
        if self._pitch is None:
            self._pitch = self.calc_pitch(self.nmom, self.scs.diry)
        return self._pitch

    def calc_yaw(self, nmom: Vector, scs_dirz: Vector) -> float:
        return nmom.dot(scs_dirz)

    @property
    def yaw(self) -> float:
        if self._yaw is None:
            self._yaw = self.calc_yaw(self.nmom, self.scs.dirz)
        return self._yaw

    @property
    def CDi(self) -> float:
        if self._CDi is None:
            self._CDi = self.dragi/self.qS
        return self._CDi

    @property
    def CD(self) -> float:
        if self._CD is None:
            self._CD = self.CDo + self.CDi
        return self._CD

    @property
    def CY(self) -> float:
        if self._CY is None:
            self._CY = self.side/self.qS
        return self._CY

    @property
    def CL(self) -> float:
        if self._CL is None:
            self._CL = self.lift/self.qS
        return self._CL

    @property
    def Cl(self) -> float:
        if self._Cl is None:
            self._Cl = self.roll/self.qS/self.bref
        return self._Cl

    @property
    def Cm(self) -> float:
        if self._Cm is None:
            self._Cm = self.pitch/self.qS/self.cref
        return self._Cm

    @property
    def Cn(self) -> float:
        if self._Cn is None:
            self._Cn = self.yaw/self.qS/self.bref
        return self._Cn

    @property
    def e(self) -> float:
        if self._e is None:
            if self.CDi == 0.0:
                if self.CL == 0.0 and self.CY == 0.0:
                    self._e = 0.0
                else:
                    self._e = float('nan')
            else:
                self._e = (self.CL**2 + self.CY**2)/pi/self.system.ar/self.CDi
        return self._e

    @property
    def lod(self) -> float:
        if self._lod is None:
            self._lod = self.CL/self.CD
        return self._lod

    def calc_Cxyz(self) -> tuple[float, float, float]:
        CDYL = Vector(self.CD, self.CY, self.CL)
        Cxyz = self.acs.vector_to_global(CDYL)
        Cx = -Cxyz.x
        Cy = Cxyz.y
        Cz = -Cxyz.z
        return Cx, Cy, Cz

    @property
    def Cx(self) -> float:
        if self._Cx is None:
            self._Cx, self._Cy, self._Cz = self.calc_Cxyz()
        return self._Cx

    @property
    def Cy(self) -> float:
        if self._Cy is None:
            self._Cx, self._Cy, self._Cz = self.calc_Cxyz()
        return self._Cy

    @property
    def Cz(self) -> float:
        if self._Cz is None:
            self._Cx, self._Cy, self._Cz = self.calc_Cxyz()
        return self._Cz


class StabilityDirectResult(DirectResult):
    dacs: TransformDerivative | None
    dscs: TransformDerivative | None

    def __init__(self, result: ConstantResult, dsig: 'NDArray',
                 dmud: 'NDArray', dmun: 'NDArray', dmuw: 'NDArray',
                 dvfs: Vector | None = None, dofs: Vector | None = None,
                 dacs: TransformDerivative | None = None,
                 dscs: TransformDerivative | None = None) -> None:
        self.dacs = dacs
        self.dscs = dscs
        super().__init__(result, dsig, dmud, dmun, dmuw, dvfs, dofs)

    @property
    def ngfrc(self) -> Vector:
        if self._ngfrc is None:
            dngvel = self.ngvel
            nlvec = self.result.result.nlvec
            self._ngfrc = self.calc_ngfrc(dngvel, nlvec)
            ngvel = self.result.result.ngvel
            dnlvec = self.nlvec
            self._ngfrc += self.calc_ngfrc(ngvel, dnlvec)
        return self._ngfrc

    @property
    def CDo(self) -> float:
        return 0.0

    @property
    def dragi(self) -> float:
        if self._dragi is None:
            dnfrc = self.nfrc
            acs_dirx = self.result.acs.dirx
            self._dragi = self.calc_dragi(dnfrc, acs_dirx)
            if self.dacs is not None:
                nfrc = self.result.result.nfrc
                dacs_dirx = self.dacs.dirx
                self._dragi += self.calc_dragi(nfrc, dacs_dirx)
        return self._dragi

    @property
    def side(self) -> float:
        if self._side is None:
            dnfrc = self.nfrc
            acs_diry = self.result.acs.diry
            self._side = self.calc_side(dnfrc, acs_diry)
            if self.dacs is not None:
                nfrc = self.result.result.nfrc
                dacs_diry = self.dacs.diry
                self._side += self.calc_side(nfrc, dacs_diry)
        return self._side

    @property
    def lift(self) -> float:
        if self._lift is None:
            dnfrc = self.nfrc
            acs_dirz = self.result.acs.dirz
            self._lift = self.calc_lift(dnfrc, acs_dirz)
            if self.dacs is not None:
                nfrc = self.result.result.nfrc
                dacs_dirz = self.dacs.dirz
                self._lift += self.calc_lift(nfrc, dacs_dirz)
        return self._lift

    @property
    def roll(self) -> float:
        if self._roll is None:
            dnmom = self.nmom
            scs_dirx = self.result.scs.dirx
            self._roll = self.calc_roll(dnmom, scs_dirx)
            if self.dscs is not None:
                nmom = self.result.result.nmom
                dscs_dirx = self.dscs.dirx
                self._roll += self.calc_roll(nmom, dscs_dirx)
        return self._roll

    @property
    def pitch(self) -> float:
        if self._pitch is None:
            dnmom = self.nmom
            scs_diry = self.result.scs.diry
            self._pitch = self.calc_pitch(dnmom, scs_diry)
            if self.dscs is not None:
                nmom = self.result.result.nmom
                dscs_diry = self.dscs.diry
                self._pitch += self.calc_pitch(nmom, dscs_diry)
        return self._pitch

    @property
    def yaw(self) -> float:
        if self._yaw is None:
            dnmom = self.nmom
            scs_dirz = self.result.scs.dirz
            self._yaw = self.calc_yaw(dnmom, scs_dirz)
            if self.dscs is not None:
                nmom = self.result.result.nmom
                dscs_dirz = self.dscs.dirz
                self._yaw += self.calc_yaw(nmom, dscs_dirz)
        return self._yaw

    def calc_Cxyz(self) -> tuple[float, float, float]:
        Cx, Cy, Cz = 0.0, 0.0, 0.0
        if self.dacs is not None:
            CD0 = self.result.result.CD
            CY0 = self.result.result.CY
            CL0 = self.result.result.CL
            avec0 = Vector(CD0, CY0, CL0)
            Cxyz0 = self.dacs.vector_to_global(avec0)
            Cx -= Cxyz0.x
            Cy += Cxyz0.y
            Cz -= Cxyz0.z
        avec = Vector(self.CD, self.CY, self.CL)
        Cxyz = self.acs.vector_to_local(avec)
        Cx -= Cxyz.x
        Cy += Cxyz.y
        Cz -= Cxyz.z
        return Cx, Cy, Cz


class StabilityResult:
    result: ConstantResult
    _u: StabilityDirectResult
    _v: StabilityDirectResult
    _w: StabilityDirectResult
    _p: StabilityDirectResult
    _q: StabilityDirectResult
    _r: StabilityDirectResult
    _alpha: StabilityDirectResult
    _beta: StabilityDirectResult
    _pbo2V: StabilityDirectResult
    _qco2V: StabilityDirectResult
    _rbo2V: StabilityDirectResult
    _pdbo2V: StabilityDirectResult
    _qdco2V: StabilityDirectResult
    _rdbo2V: StabilityDirectResult
    _xnp: float
    _sprat: float

    __slots__ = tuple(__annotations__)

    def __init__(self, result: ConstantResult) -> None:
        self.result = result
        self.reset()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def u(self) -> StabilityDirectResult:
        if self._u is None:
            dvfs = Vector(1.0, 0.0, 0.0)*self.result.speed
            dsig = self.result.calculate_sig(vfs = dvfs)
            dmud = self.result.calculate_mud(vfs = dvfs)
            dmun = self.result.calculate_mun(vfs = dvfs)
            dmuw = self.result.calculate_muw(vfs = dvfs)
            self._u = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                            dmuw, dvfs = dvfs)
        return self._u

    @property
    def v(self) -> StabilityDirectResult:
        if self._v is None:
            dvfs = Vector(0.0, -1.0, 0.0)*self.result.speed
            dsig = self.result.calculate_sig(vfs = dvfs)
            dmud = self.result.calculate_mud(vfs = dvfs)
            dmun = self.result.calculate_mun(vfs = dvfs)
            dmuw = self.result.calculate_muw(vfs = dvfs)
            self._v = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                            dmuw, dvfs = dvfs)
        return self._v

    @property
    def w(self) -> StabilityDirectResult:
        if self._w is None:
            dvfs = Vector(0.0, 0.0, 1.0)*self.result.speed
            dsig = self.result.calculate_sig(vfs = dvfs)
            dmud = self.result.calculate_mud(vfs = dvfs)
            dmun = self.result.calculate_mun(vfs = dvfs)
            dmuw = self.result.calculate_muw(vfs = dvfs)
            self._w = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                            dmuw, dvfs = dvfs)
        return self._w

    @property
    def p(self) -> StabilityDirectResult:
        if self._p is None:
            dpqr = Vector(1.0, 0.0, 0.0)
            dofs = self.result.scs.vector_to_global(dpqr)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._p = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                            dmuw, dofs = dofs)
        return self._p

    @property
    def q(self) -> StabilityDirectResult:
        if self._q is None:
            dpqr = Vector(0.0, 1.0, 0.0)
            dofs = self.result.scs.vector_to_global(dpqr)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._q = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                            dmuw, dofs = dofs)
        return self._q

    @property
    def r(self) -> StabilityDirectResult:
        if self._r is None:
            dpqr = Vector(0.0, 0.0, 1.0)
            dofs = self.result.scs.vector_to_global(dpqr)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._r = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                            dmuw, dofs = dofs)
        return self._r

    @property
    def alpha(self) -> StabilityDirectResult:
        if self._alpha is None:
            dvfs = self.result.dvfsa
            dofs = self.result.dscsa.vector_to_global(self.result.pqr)
            dsig = self.result.calculate_sig(vfs = dvfs, ofs = dofs)
            dmud = self.result.calculate_mud(vfs = dvfs, ofs = dofs)
            dmun = self.result.calculate_mun(vfs = dvfs, ofs = dofs)
            dmuw = self.result.calculate_muw(vfs = dvfs, ofs = dofs)
            self._alpha = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                dmuw, dvfs = dvfs, dofs = dofs,
                                                dacs = self.result.dacsa,
                                                dscs = self.result.dscsa)
        return self._alpha

    @property
    def beta(self) -> StabilityDirectResult:
        if self._beta is None:
            dvfs = self.result.dvfsb
            dsig = self.result.calculate_sig(vfs = dvfs)
            dmud = self.result.calculate_mud(vfs = dvfs)
            dmun = self.result.calculate_mun(vfs = dvfs)
            dmuw = self.result.calculate_muw(vfs = dvfs)
            self._beta = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                               dmuw, dvfs = dvfs)
        return self._beta

    @property
    def pbo2V(self) -> StabilityDirectResult:
        if self._pbo2V is None:
            dp = 2*self.result.speed/self.result.system.bref
            dpqr = Vector(dp, 0.0, 0.0)
            dofs = self.result.scs.vector_to_global(dpqr)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._pbo2V = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                dmuw, dofs = dofs)
        return self._pbo2V

    @property
    def qco2V(self) -> StabilityDirectResult:
        if self._qco2V is None:
            dq = 2*self.result.speed/self.result.system.cref
            dpqr = Vector(0.0, dq, 0.0)
            dofs = self.result.scs.vector_to_global(dpqr)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._qco2V = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                dmuw, dofs = dofs)
        return self._qco2V

    @property
    def rbo2V(self) -> StabilityDirectResult:
        if self._rbo2V is None:
            dr = 2*self.result.speed/self.result.system.bref
            dpqr = Vector(0.0, 0.0, dr)
            dofs = self.result.scs.vector_to_global(dpqr)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._rbo2V = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                dmuw, dofs = dofs)
        return self._rbo2V

    @property
    def pdbo2V(self) -> StabilityDirectResult:
        if self._pdbo2V is None:
            dpd = 2*self.result.speed/self.result.system.bref
            dofs = Vector(dpd, 0.0, 0.0)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._pdbo2V = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                 dmuw, dofs = dofs)
        return self._pdbo2V

    @property
    def qdco2V(self) -> StabilityDirectResult:
        if self._qdco2V is None:
            dqd = 2*self.result.speed/self.result.system.cref
            dofs = Vector(0.0, dqd, 0.0)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._qdco2V = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                 dmuw, dofs = dofs)
        return self._qdco2V

    @property
    def rdbo2V(self) -> StabilityDirectResult:
        if self._rdbo2V is None:
            drd = 2*self.result.speed/self.result.system.bref
            dofs = Vector(0.0, 0.0, drd)
            dsig = self.result.calculate_sig(ofs = dofs)
            dmud = self.result.calculate_mud(ofs = dofs)
            dmun = self.result.calculate_mun(ofs = dofs)
            dmuw = self.result.calculate_muw(ofs = dofs)
            self._rdbo2V = StabilityDirectResult(self.result, dsig, dmud, dmun,
                                                 dmuw, dofs = dofs)
        return self._rdbo2V

    @property
    def xnp(self) -> float:
        if self._xnp is None:
            xcg = self.result.rcg.x
            CLa = self.alpha.CL
            CMa = self.alpha.Cm
            c = self.result.system.cref
            self._xnp = xcg - c*CMa/CLa
        return self._xnp

    @property
    def sprat(self) -> float:
        if self._sprat is None:
            Clb = self.beta.Cl
            Cnb = self.beta.Cn
            Cnr = self.rbo2V.Cn
            Clr = self.rbo2V.Cl
            self._sprat = Clb*Cnr/(Clr*Cnb)
        return self._sprat

    @property
    def stability_derivatives(self) -> MDReport:
        from . import sfrm
        report = MDReport()
        report.add_heading('Stability Derivatives', 2)
        table = report.add_table()
        table.add_column('C<sub>La</sub>', sfrm, data=[self.alpha.CL])
        table.add_column('C<sub>Ya</sub>', sfrm, data=[self.alpha.CY])
        table.add_column('C<sub>la</sub>', sfrm, data=[self.alpha.Cl])
        table.add_column('C<sub>ma</sub>', sfrm, data=[self.alpha.Cm])
        table.add_column('C<sub>na</sub>', sfrm, data=[self.alpha.Cn])
        table = report.add_table()
        table.add_column('C<sub>Lb</sub>', sfrm, data=[self.beta.CL])
        table.add_column('C<sub>Yb</sub>', sfrm, data=[self.beta.CY])
        table.add_column('C<sub>lb</sub>', sfrm, data=[self.beta.Cl])
        table.add_column('C<sub>mb</sub>', sfrm, data=[self.beta.Cm])
        table.add_column('C<sub>nb</sub>', sfrm, data=[self.beta.Cn])
        table = report.add_table()
        table.add_column('C<sub>Lp</sub>', sfrm, data=[self.pbo2V.CL])
        table.add_column('C<sub>Yp</sub>', sfrm, data=[self.pbo2V.CY])
        table.add_column('C<sub>lp</sub>', sfrm, data=[self.pbo2V.Cl])
        table.add_column('C<sub>mp</sub>', sfrm, data=[self.pbo2V.Cm])
        table.add_column('C<sub>np</sub>', sfrm, data=[self.pbo2V.Cn])
        table = report.add_table()
        table.add_column('C<sub>Lq</sub>', sfrm, data=[self.qco2V.CL])
        table.add_column('C<sub>Yq</sub>', sfrm, data=[self.qco2V.CY])
        table.add_column('C<sub>lq</sub>', sfrm, data=[self.qco2V.Cl])
        table.add_column('C<sub>mq</sub>', sfrm, data=[self.qco2V.Cm])
        table.add_column('C<sub>nq</sub>', sfrm, data=[self.qco2V.Cn])
        table = report.add_table()
        table.add_column('C<sub>Lr</sub>', sfrm, data=[self.rbo2V.CL])
        table.add_column('C<sub>Yr</sub>', sfrm, data=[self.rbo2V.CY])
        table.add_column('C<sub>lr</sub>', sfrm, data=[self.rbo2V.Cl])
        table.add_column('C<sub>mr</sub>', sfrm, data=[self.rbo2V.Cm])
        table.add_column('C<sub>nr</sub>', sfrm, data=[self.rbo2V.Cn])
        report.add_heading(f'Neutral Point Xnp = {self.xnp:.6f}', 3)
        spratstr = 'C<sub>lb</sub>.C<sub>nr</sub>/(C<sub>lr</sub>.C<sub>nb</sub>)'
        report.add_heading(f'{spratstr:s} = {self.sprat:.6f} (> 1 if spirally stable)', 3)
        return report

    @property
    def stability_derivatives_body(self) -> MDReport:
        from . import sfrm
        report = MDReport()
        report.add_heading('Stability Derivatives Body Axis', 2)
        table = report.add_table()
        table.add_column('C<sub>xu</sub>', sfrm, data=[self.u.Cx])
        table.add_column('C<sub>yu</sub>', sfrm, data=[self.u.Cy])
        table.add_column('C<sub>zu</sub>', sfrm, data=[self.u.Cz])
        table.add_column('C<sub>lu</sub>', sfrm, data=[self.u.Cl])
        table.add_column('C<sub>mu</sub>', sfrm, data=[self.u.Cm])
        table.add_column('C<sub>nu</sub>', sfrm, data=[self.u.Cn])
        table = report.add_table()
        table.add_column('C<sub>xv</sub>', sfrm, data=[self.v.Cx])
        table.add_column('C<sub>yv</sub>', sfrm, data=[self.v.Cy])
        table.add_column('C<sub>zv</sub>', sfrm, data=[self.v.Cz])
        table.add_column('C<sub>lv</sub>', sfrm, data=[self.v.Cl])
        table.add_column('C<sub>mv</sub>', sfrm, data=[self.v.Cm])
        table.add_column('C<sub>nv</sub>', sfrm, data=[self.v.Cn])
        table = report.add_table()
        table.add_column('C<sub>xw</sub>', sfrm, data=[self.w.Cx])
        table.add_column('C<sub>yw</sub>', sfrm, data=[self.w.Cy])
        table.add_column('C<sub>zw</sub>', sfrm, data=[self.w.Cz])
        table.add_column('C<sub>lw</sub>', sfrm, data=[self.w.Cl])
        table.add_column('C<sub>mw</sub>', sfrm, data=[self.w.Cm])
        table.add_column('C<sub>nw</sub>', sfrm, data=[self.w.Cn])
        table = report.add_table()
        table.add_column('C<sub>xp</sub>', sfrm, data=[self.pbo2V.Cx])
        table.add_column('C<sub>yp</sub>', sfrm, data=[self.pbo2V.Cy])
        table.add_column('C<sub>zp</sub>', sfrm, data=[self.pbo2V.Cz])
        table.add_column('C<sub>lp</sub>', sfrm, data=[self.pbo2V.Cl])
        table.add_column('C<sub>mp</sub>', sfrm, data=[self.pbo2V.Cm])
        table.add_column('C<sub>np</sub>', sfrm, data=[self.pbo2V.Cn])
        table = report.add_table()
        table.add_column('C<sub>xq</sub>', sfrm, data=[self.qco2V.Cx])
        table.add_column('C<sub>yq</sub>', sfrm, data=[self.qco2V.Cy])
        table.add_column('C<sub>zq</sub>', sfrm, data=[self.qco2V.Cz])
        table.add_column('C<sub>lq</sub>', sfrm, data=[self.qco2V.Cl])
        table.add_column('C<sub>mq</sub>', sfrm, data=[self.qco2V.Cm])
        table.add_column('C<sub>nq</sub>', sfrm, data=[self.qco2V.Cn])
        table = report.add_table()
        table.add_column('C<sub>xr</sub>', sfrm, data=[self.rbo2V.Cx])
        table.add_column('C<sub>yr</sub>', sfrm, data=[self.rbo2V.Cy])
        table.add_column('C<sub>zr</sub>', sfrm, data=[self.rbo2V.Cz])
        table.add_column('C<sub>lr</sub>', sfrm, data=[self.rbo2V.Cl])
        table.add_column('C<sub>mr</sub>', sfrm, data=[self.rbo2V.Cm])
        table.add_column('C<sub>nr</sub>', sfrm, data=[self.rbo2V.Cn])
        return report


class StabilityModes:
    result: ConstantResult
    _Asys: 'NDArray'
    _eigvals: 'NDArray'
    _eigvecs: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, result: ConstantResult) -> None:
        self.result = result
        self.reset()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def Asys(self) -> 'NDArray':
        if self._Asys is None:
            b = self.result.system.bref
            c = self.result.system.cref
            S = self.result.system.sref
            m = self.result.mass.mass
            I = self.result.mass.Imat
            Q0 = self.result.qfs
            rho = self.result.rho
            u0 = self.result.vfs.x
            v0 = self.result.vfs.y
            w0 = self.result.vfs.z
            V0 = self.result.speed
            b_2V0 = b/2/V0
            c_2V0 = c/2/V0
            QS0 = Q0*S
            QS0c = QS0*c
            QS0b = QS0*b
            Cx0 = self.result.result.Cx
            Cy0 = self.result.result.Cy
            Cz0 = self.result.result.Cz
            Cl0 = self.result.result.Cl
            Cm0 = self.result.result.Cm
            Cn0 = self.result.result.Cn

            Cxu = self.result.stab_der_res.u.Cx/V0
            Cxv = self.result.stab_der_res.v.Cx/V0
            Cxw = self.result.stab_der_res.w.Cx/V0
            Cxp = self.result.stab_der_res.pbo2V.Cx*b_2V0
            Cxq = self.result.stab_der_res.qco2V.Cx*c_2V0
            Cxr = self.result.stab_der_res.rbo2V.Cx*b_2V0
            Cyu = self.result.stab_der_res.u.Cy/V0
            Cyv = self.result.stab_der_res.v.Cy/V0
            Cyw = self.result.stab_der_res.w.Cy/V0
            Cyp = self.result.stab_der_res.pbo2V.Cy*b_2V0
            Cyq = self.result.stab_der_res.qco2V.Cy*c_2V0
            Cyr = self.result.stab_der_res.rbo2V.Cy*b_2V0
            Czu = self.result.stab_der_res.u.Cz/V0
            Czv = self.result.stab_der_res.v.Cz/V0
            Czw = self.result.stab_der_res.w.Cz/V0
            Czp = self.result.stab_der_res.pbo2V.Cz*b_2V0
            Czq = self.result.stab_der_res.qco2V.Cz*c_2V0
            Czr = self.result.stab_der_res.rbo2V.Cz*b_2V0
            Clu = self.result.stab_der_res.u.Cl/V0
            Clv = self.result.stab_der_res.v.Cl/V0
            Clw = self.result.stab_der_res.w.Cl/V0
            Clp = self.result.stab_der_res.pbo2V.Cl*b_2V0
            Clq = self.result.stab_der_res.qco2V.Cl*c_2V0
            Clr = self.result.stab_der_res.rbo2V.Cl*b_2V0
            Cmu = self.result.stab_der_res.u.Cm/V0
            Cmv = self.result.stab_der_res.v.Cm/V0
            Cmw = self.result.stab_der_res.w.Cm/V0
            Cmp = self.result.stab_der_res.pbo2V.Cm*b_2V0
            Cmq = self.result.stab_der_res.qco2V.Cm*c_2V0
            Cmr = self.result.stab_der_res.rbo2V.Cm*b_2V0
            Cnu = self.result.stab_der_res.u.Cn/V0
            Cnv = self.result.stab_der_res.v.Cn/V0
            Cnw = self.result.stab_der_res.w.Cn/V0
            Cnp = self.result.stab_der_res.pbo2V.Cn*b_2V0
            Cnq = self.result.stab_der_res.qco2V.Cn*c_2V0
            Cnr = self.result.stab_der_res.rbo2V.Cn*b_2V0

            Xu = QS0*Cxu + Cx0*S*rho*u0
            Xv = QS0*Cxv + Cx0*S*rho*v0
            Xw = QS0*Cxw + Cx0*S*rho*w0
            Xp = QS0*Cxp
            Xq = QS0*Cxq
            Xr = QS0*Cxr
            Yu = QS0*Cyu + Cy0*S*rho*u0
            Yv = QS0*Cyv + Cy0*S*rho*v0
            Yw = QS0*Cyw + Cy0*S*rho*w0
            Yp = QS0*Cyp
            Yq = QS0*Cyq
            Yr = QS0*Cyr
            Zu = QS0*Czu + Cz0*S*rho*u0
            Zv = QS0*Czv + Cz0*S*rho*v0
            Zw = QS0*Czw + Cz0*S*rho*w0
            Zp = QS0*Czp
            Zq = QS0*Czq
            Zr = QS0*Czr
            Lu = QS0b*Clu + Cl0*S*rho*u0
            Lv = QS0b*Clv + Cl0*S*rho*v0
            Lw = QS0b*Clw + Cl0*S*rho*w0
            Lp = QS0b*Clp
            Lq = QS0b*Clq
            Lr = QS0b*Clr
            Mu = QS0c*Cmu + Cm0*S*rho*u0
            Mv = QS0c*Cmv + Cm0*S*rho*v0
            Mw = QS0c*Cmw + Cm0*S*rho*w0
            Mp = QS0c*Cmp
            Mq = QS0c*Cmq
            Mr = QS0c*Cmr
            Nu = QS0b*Cnu + Cn0*S*rho*u0
            Nv = QS0b*Cnv + Cn0*S*rho*v0
            Nw = QS0b*Cnw + Cn0*S*rho*w0
            Np = QS0b*Cnp
            Nq = QS0b*Cnq
            Nr = QS0b*Cnr
            g = GRAVACC
            th0 = 0.0

            Als_XYZ = asarray([[Xu, Xv, Xw, Xp, Xq, Xr],
                               [Yu, Yv, Yw, Yp, Yq, Yr],
                               [Zu, Zv, Zw, Zp, Zq, Zr]])

            Als_LMN = asarray([[Lu, Lv, Lw, Lp, Lq, Lr],
                               [Mu, Mv, Mw, Mp, Mq, Mr],
                               [Nu, Nv, Nw, Np, Nq, Nr]])

            Als_XYZ_m = Als_XYZ/m
            Als_LMN_I = solve(I, Als_LMN)

            Als = zeros((8, 8))
            Als[0, 3] = -g*cos(th0)
            Als[1, 2] = V0
            Als[1, 3] = -g*sin(th0)
            Als[3, 2] = 1.0
            Als[4, 6] = -V0
            Als[4, 7] = g*cos(th0)
            Als[7, 5] = 1.0
            Als[(0, 1, 4), 0:3] += Als_XYZ_m[(0, 2, 1), 0::2]
            Als[(0, 1, 4), 4:7] += Als_XYZ_m[(0, 2, 1), 1::2]
            Als[(2, 5, 6), 0:3] += Als_LMN_I[(1, 0, 2), 0::2]
            Als[(2, 5, 6), 4:7] += Als_LMN_I[(1, 0, 2), 1::2]

            self._Asys = Als

        return self._Asys

    @property
    def eigvals(self) -> 'NDArray':
        if self._eigvals is None:
            self._eigvals, self._eigvecs = eig(self.Asys)
        return self._eigvals

    @property
    def eigvecs(self) -> 'NDArray':
        if self._eigvecs is None:
            self._eigvals, self._eigvecs = eig(self.Asys)
        return self._eigvecs

    def stability_modes(self, sfrm: str | None =  None) -> MDReport:
        if sfrm is None:
            from . import sfrm
        report = MDReport()
        report.add_heading('Stability Modes', 2)
        report.add_matrix('A_{sys}', self.Asys, sfrm)
        report.add_heading('Eigenvalues', 2)
        table = report.add_table()
        table.add_column('Real', sfrm, data=self.eigvals.real.tolist())
        table.add_column('Imag', sfrm, data=self.eigvals.imag.tolist())
        return report

class ControlResult(DirectResult):

    ctrl_obj: 'ControlObject'
    _hprel: Vector | None
    _hpmom: Vector | None
    _hmom: float | None
    _Ch: float | None

    def __init__(self, result: ConstantResult, ctrl_obj: 'ControlObject',
                 mud: 'NDArray', mun: 'NDArray', muw: 'NDArray') -> None:
        self.ctrl_obj = ctrl_obj
        super().__init__(result, None, mud, mun, muw)
        self._hprel = None
        self._hpmom = None
        self._hmom = None
        self._Ch = None

    @property
    def CDo(self) -> float:
        return 0.0

    @property
    def ngfrc(self) -> Vector:
        if self._ngfrc is None:
            index = self.ctrl_obj.grid_index
            self._ngfrc = Vector.zeros(self.result.result.ngfrc.shape)
            self._ngfrc[index] = self.result.result.ngfrc[index]
        return self._ngfrc

    @property
    def hprel(self) -> Vector:
        if self._hprel is None:
            self._hprel = self.system.gridvec - self.ctrl_obj.point
        return self._hprel

    @property
    def hpmom(self) -> Vector:
        if self._hpmom is None:
            self._hpmom = self.hprel.cross(self.ngfrc)
        return self._hpmom

    @property
    def hmom(self) -> Vector:
        if self._hmom is None:
            self._hmom = self.hpmom.sum().dot(self.ctrl_obj.vector)
        return self._hmom

    @property
    def Ch(self) -> float:
        if self._Ch is None:
            self._Ch = self.hmom/self.qS/self.cref
        return self._Ch


class TrimResult:
    result: ConstantResult
    targets: dict[str, tuple[str, float]]
    initstate: dict[str, float]
    initctrls: dict[str, float]

    __slots__ = tuple(__annotations__)

    def __init__(self, result: ConstantResult) -> None:
        self.result = result
        self.targets = {
            'alpha': ('alpha', 0.0),
            'beta': ('beta', 0.0),
            'pbo2V': ('pbo2V', 0.0),
            'qco2V': ('qco2V', 0.0),
            'rbo2V': ('rbo2V', 0.0),
        }
        for control in self.result.ctrls:
            self.targets[control] = (control, 0.0)

    def set_state(self, mach: float | None = None, speed: float | None = None,
                  alpha: float | None = None, beta: float | None = None,
                  pbo2V: float | None = None, qco2V: float | None = None,
                  rbo2V: float | None = None) -> None:
        self.result.set_state(mach=mach, speed=speed, alpha=alpha, beta=beta,
                              pbo2V=pbo2V, qco2V=qco2V, rbo2V=rbo2V)
        if self.targets['alpha'][0] == 'alpha':
            self.targets['alpha'] = ('alpha', self.result.alpha)
        if self.targets['beta'][0] == 'beta':
            self.targets['beta'] = ('beta', self.result.beta)
        if self.targets['pbo2v'][0] == 'pbo2v':
            self.targets['pbo2v'] = ('pbo2v', self.result.pbo2v)
        if self.targets['qco2v'][0] == 'qco2v':
            self.targets['qco2v'] = ('qco2v', self.result.qco2v)
        if self.targets['rbo2v'][0] == 'rbo2v':
            self.targets['rbo2v'] = ('rbo2v', self.result.rbo2v)

    def set_controls(self, **kwargs: dict[str, float]) -> None:
        self.result.set_controls(**kwargs)
        for control in kwargs:
            if control in self.targets:
                self.targets[control] = (control, kwargs[control])

    def set_targets(self, CLt: float | None = None, CYt: float | None = None,
                    Clt: float | None = None, Cmt: float | None = None,
                    Cnt: float | None = None) -> None:
        if CLt is not None:
            self.targets['alpha'] = ('CL', CLt)
        if CYt is not None:
            self.targets['beta'] = ('CY', CYt)
        controls = self.result.ctrls.keys()
        moment = {}
        if Clt is not None:
            moment['Cl'] = Clt
        if Cmt is not None:
            moment['Cm'] = Cmt
        if Cnt is not None:
            moment['Cn'] = Cnt
        for control, (name, value) in zip(controls, moment.items()):
            self.targets[control] = (name, value)

    def set_initial_state(self, initstate: dict[str, float]) -> None:
        if 'alpha' in self.targets:
            if self.targets['alpha'][0] == 'alpha':
                initstate['alpha'] = self.targets['alpha'][1]
        if 'beta' in self.targets:
            if self.targets['beta'][0] == 'beta':
                initstate['beta'] = self.targets['beta'][1]
        if 'pbo2V' in self.targets:
            if self.targets['pbo2V'][0] == 'pbo2V':
                initstate['pbo2V'] = self.targets['pbo2V'][1]
        if 'qco2V' in self.targets:
            if self.targets['qco2V'][0] == 'qco2V':
                initstate['qco2V'] = self.targets['qco2V'][1]
        if 'rbo2V' in self.targets:
            if self.targets['rbo2V'][0] == 'rbo2V':
                initstate['rbo2V'] = self.targets['rbo2V'][1]
        self.initstate = initstate
        self.result.set_state(**self.initstate)

    def set_initial_controls(self, initctrls: dict[str, float]) -> None:
        for control in self.result.ctrls:
            if control in self.targets:
                if self.targets[control][0] == control:
                    initctrls[control] = self.targets[control][1]
        self.initctrls = initctrls
        self.result.set_controls(**self.initctrls)

    def target_Cmat(self) -> 'NDArray':
        Ctgt = zeros(len(self.targets))
        for i, value in enumerate(self.targets.values()):
            if value[0] == 'alpha':
                Ctgt[i] = radians(value[1])
            elif value[0] == 'beta':
                Ctgt[i] = radians(value[1])
            elif value[0] in self.result.ctrls:
                Ctgt[i] = radians(value[1])
            else:
                Ctgt[i] = value[1]
        return Ctgt

    def current_Cmat(self) -> 'NDArray':
        Ccur = zeros(len(self.targets))
        for i, value in enumerate(self.targets.values()):
            if hasattr(self.result, value[0].lower()):
                current_value = getattr(self.result, value[0].lower())
                if value[0] == 'alpha':
                    Ccur[i] = radians(current_value)
                elif value[0] == 'beta':
                    Ccur[i] = radians(current_value)
                else:
                    Ccur[i] = current_value
            elif hasattr(self.result.result, value[0]):
                Ccur[i] = getattr(self.result.result, value[0])
            elif value[0] in self.result.ctrls:
                current_value = self.result.ctrls[value[0]]
                Ccur[i] = radians(current_value)
            else:
                raise ValueError(f'Unknown target {value[0]}')
        return Ccur

    def delta_C(self) -> 'NDArray':
        Ctgt = self.target_Cmat()
        Ccur = self.current_Cmat()
        return Ctgt - Ccur

    def current_Dmat(self) -> 'NDArray':
        Dcur = zeros(len(self.targets))
        for i, variable in enumerate(self.targets):
            if variable == 'alpha':
                Dcur[i] = radians(self.result.alpha)
            elif variable == 'beta':
                Dcur[i] = radians(self.result.beta)
            elif variable == 'pbo2v':
                Dcur[i] = self.result.pbo2v
            elif variable == 'qco2v':
                Dcur[i] = self.result.qco2v
            elif variable == 'rbo2v':
                Dcur[i] = self.result.rbo2v
            elif variable in self.result.ctrls:
                Dcur[i] = radians(self.result.ctrls[variable])
        return Dcur

    def current_Hmat(self) -> 'NDArray':
        num = len(self.targets)
        Hcur = zeros((num, num))
        for i, target in enumerate(self.targets.values()):
            for j, variable in enumerate(self.targets):
                if variable == target[0]:
                    Hcur[i, :] = 0.0
                    Hcur[i, j] = 1.0
                else:
                    if hasattr(self.result.stab_der_res, variable):
                        stvar = getattr(self.result.stab_der_res, variable)
                        if hasattr(stvar, target[0]):
                            Hcur[i, j] = getattr(stvar, target[0])
                    elif variable in self.result.ctrl_der_res:
                        ctvar = self.result.ctrl_der_res[variable]
                        if hasattr(ctvar, target[0]):
                            Hcur[i, j] = getattr(ctvar, target[0])
        return Hcur

    def trim(self, crit: float = 1e-6, imax: int = 100, display: bool = False) -> None:
        count = 0
        while True:
            Cdff = self.delta_C()
            nrmC = norm(Cdff)
            if nrmC < crit:
                break
            Hcur = self.current_Hmat()
            Ddff = solve(Hcur, Cdff)
            Dcur = self.current_Dmat() + Ddff
            if display:
                if count == 0:
                    outstr = f'\nTrimming {self.result.name}\n'
                    outstr += f'{"iter":>4s}  {"d(alpha)":>11s}  {"d(beta)":>11s}'
                    outstr += f'  {"d(pbo2V)":>11s}  {"d(qco2V)":>11s}  {"d(rbo2V)":>11s}'
                    for control in self.result.ctrls:
                        control = f'd({control})'
                        outstr += f'  {control:>11s}'
                    print(outstr)
                outstr = f'{count+1:4d}'
                for i, variable in enumerate(self.targets):
                    outstr += f'  {Ddff[i]:11.3e}'
                print(outstr)
            state = {}
            ctrls = {}
            for i, variable in enumerate(self.targets):
                if variable == 'alpha':
                    state['alpha'] = degrees(Dcur[i])
                elif variable == 'beta':
                    state['beta'] = degrees(Dcur[i])
                elif variable == 'pbo2V':
                    state['pbo2V'] = Dcur[i]
                elif variable == 'qco2V':
                    state['qco2V'] = Dcur[i]
                elif variable == 'rbo2V':
                    state['rbo2V'] = Dcur[i]
                elif variable in self.result.ctrls:
                    ctrls[variable] = degrees(Dcur[i])

            self.result.set_state(**state)
            self.result.set_controls(**ctrls)
            count += 1
            if count >= imax:
                print(f'Convergence failed for {self.result.name}.')
                return False


class StripResult:
    result: ConstantResult
    surface: 'ConstantSurface'
    _strip_forces: Vector
    _strip_distribution: Vector

    __slots__ = tuple(__annotations__)

    def __init__(self, result: ConstantResult,
                 surface: 'ConstantSurface') -> None:
        self.result = result
        self.surface = surface
        self._strip_forces = None
        self._strip_distribution = None

    @property
    def strip_forces(self) -> Vector:
        if self._strip_forces is None:
            grid_forces = self.result.result.ngfrc[self.surface.ngridindex]
            self._strip_forces = grid_forces.sum(axis=1)
        return self._strip_forces

    @property
    def strip_distribution(self) -> Vector:
        if self._strip_distribution is None:
            self._strip_distribution = self.strip_forces/self.surface.w
        return self._strip_distribution

    def plot_strip_distribution(self, ax: 'Axes' = None,
                                **kwargs: dict[str, Any]) -> 'Axes':

        xvar = kwargs.pop('xvar', 'b')
        if xvar == 'b':
            x = self.surface.b
            xlabel = 'Span (m)'
        elif xvar == 'y':
            x = self.surface.y
            xlabel = 'Y (m)'
        elif xvar == 'z':
            x = self.surface.z
            xlabel = 'Z (m)'
        else:
            raise ValueError('Unknown xvar.')

        yvar = kwargs.pop('yvar', 'l')
        if yvar == 'l':
            y = self.strip_distribution.dot(self.result.acs.dirz)
            ylabel = 'l (N/m)'
        elif yvar == 'd':
            y = self.strip_distribution.dot(self.result.acs.dirx)
            ylabel = 'd (N/m)'
        elif yvar == 'y':
            y = self.strip_distribution.dot(self.result.acs.diry)
            ylabel = 'y (N/m)'
        else:
            raise ValueError('Unknown yvar.')

        if ax is None:
            figsize = kwargs.pop('figsize', (12, 8))
            fig = figure(figsize=figsize)
            ax = fig.gca()
            ax.grid(True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        kwargs.setdefault('label', self.surface.name)
        ax.plot(x, y, **kwargs)
        ax.legend()
        return ax

    def __repr__(self) -> str:
        return f'{self.surface.name} strip result for {self.result.name}'


def trig_angle(angle: float) -> float:
    '''Calculates cos(angle) and sin(angle) with angle in degrees.'''
    angrad = radians(angle)
    cosang = cos(angrad)
    sinang = sin(angrad)
    return cosang, sinang
