from time import perf_counter
from typing import TYPE_CHECKING, Dict, List

from numpy import degrees, radians, zeros
from numpy.linalg import inv, norm

from ..tools.mass import Mass
from ..tools.trim import LoadTrim, LoopingTrim, TurningTrim
from .panelresult import NearFieldResult, PanelResult

ANGTOL = 30.0

if TYPE_CHECKING:
    from .panelsystem import PanelSystem

class PanelTrim(PanelResult):
    CLt: float = None
    CYt: float = None
    Clt: float = None
    Cmt: float = None
    Cnt: float = None
    initstate: Dict[str, float] = None
    initctrls: Dict[str, float] = None
    _tgtlst: List[str] = None
    _numtgt: int = None
    _trmmom: bool = None
    # trmfrc = None
    # trmmom = None
    # trmlft = None

    def __init__(self, name: str, sys: object):
        super().__init__(name, sys)
        # self.set_trim_loads()

    def set_targets(self, CLt: float=0.0, CYt: float=0.0,
                    Clt: float=0.0, Cmt: float=0.0, Cnt: float=0.0):
        self.CLt = CLt
        self.CYt = CYt
        self.Clt = Clt
        self.Cmt = Cmt
        self.Cnt = Cnt
        self._tgtlst = None
        self._numtgt = None
        self._trmmom = None

    def set_initial_state(self, initstate: Dict[str, float]):
        self.initstate = initstate
        # if 'alpha' not in self.initstate:
        #     self.initstate['alpha'] = 0.0
        # if 'beta' not in self.initstate:
        #     self.initstate['beta'] = 0.0
        self.set_state(**self.initstate)

    def set_initial_controls(self, initctrls: Dict[str, float]):
        self.initctrls = initctrls
        self.set_controls(**self.initctrls)

    @property
    def tgtlst(self):
        if self._tgtlst is None:
            self._tgtlst = []
            if self.CLt is not None:
                self._tgtlst.append('CL')
            if self.CYt is not None:
                self._tgtlst.append('CY')
            if self.Clt is not None:
                self._tgtlst.append('Cl')
            if self.Cmt is not None:
                self._tgtlst.append('Cm')
            if self.Cnt is not None:
                self._tgtlst.append('Cn')
        return self._tgtlst

    @property
    def trmmom(self):
        if self._trmmom is None:
            self._trmmom = False
            if 'Cl' in self.tgtlst:
                self._trmmom = True
            elif 'Cm' in self.tgtlst:
                self._trmmom = True
            elif 'Cn' in self.tgtlst:
                self._trmmom = True
        return self._trmmom

    # def set_trim_loads(self, trmfrc: bool=True, trmmom: bool=True, trmlft: bool=False):
    #     self.trmfrc = trmfrc
    #     self.trmmom = trmmom
    #     self.trmlft = trmlft

    def delta_C(self):
        Ctgt = self.target_Cmat()
        Ccur = self.current_Cmat()
        return Ctgt - Ccur

    def target_Cmat(self):
        numtgt = len(self.tgtlst)
        Ctgt = zeros((numtgt, 1), dtype=float)
        for i, tgt in enumerate(self.tgtlst):
            Ctgt[i, 0] = getattr(self, tgt + 't')
        # if self.trmlft:
        #     Ctgt = zeros((1, 1), dtype=float)
        #     Ctgt[0, 0] = self.CLt
        # elif self.trmfrc and self.trmmom:
        #     Ctgt = zeros((5, 1), dtype=float)
        #     Ctgt[0, 0] = self.CLt
        #     Ctgt[1, 0] = self.CYt
        #     Ctgt[2, 0] = self.Clt
        #     Ctgt[3, 0] = self.Cmt
        #     Ctgt[4, 0] = self.Cnt
        # elif self.trmfrc:
        #     Ctgt = zeros((2, 1), dtype=float)
        #     Ctgt[0, 0] = self.CLt
        #     Ctgt[1, 0] = self.CYt
        # elif self.trmmom:
        #     Ctgt = zeros((3, 1), dtype=float)
        #     Ctgt[0, 0] = self.Clt
        #     Ctgt[1, 0] = self.Cmt
        #     Ctgt[2, 0] = self.Cnt
        # else:
        #     Ctgt = zeros((0, 1), dtype=float)
        return Ctgt

    def current_Cmat(self):
        numtgt = len(self.tgtlst)
        Ccur = zeros((numtgt, 1), dtype=float)
        self._nfres = None
        for i, tgt in enumerate(self.tgtlst):
            Ccur[i, 0] = getattr(self.nfres, tgt)
        # if self.trmlft:
        #     Ccur = zeros((1, 1), dtype=float)
        #     Ccur[0, 0] = self.nfres.CL
        # elif self.trmfrc and self.trmmom:
        #     Ccur = zeros((5, 1), dtype=float)
        #     Ccur[0, 0] = self.nfres.CL
        #     Ccur[1, 0] = self.nfres.CY
        #     Ccur[2, 0] = self.nfres.Cl
        #     Ccur[3, 0] = self.nfres.Cm
        #     Ccur[4, 0] = self.nfres.Cn
        #     # if self.sys.cdo != 0.0:
        #     #     Ccur[0, 0] += self.pdres.CL
        #     #     Ccur[1, 0] += self.pdres.CY
        #     #     Ccur[2, 0] += self.pdres.Cl
        #     #     Ccur[3, 0] += self.pdres.Cm
        #     #     Ccur[4, 0] += self.pdres.Cn
        # elif self.trmfrc:
        #     Ccur = zeros((2, 1), dtype=float)
        #     Ccur[0, 0] = self.nfres.CL
        #     Ccur[1, 0] = self.nfres.CY
        #     # if self.sys.cdo != 0.0:
        #     #     Ccur[0, 0] += self.pdres.CL
        #     #     Ccur[1, 0] += self.pdres.CY
        # elif self.trmmom:
        #     Ccur = zeros((3, 1), dtype=float)
        #     Ccur[0, 0] = self.nfres.Cl
        #     Ccur[1, 0] = self.nfres.Cm
        #     Ccur[2, 0] = self.nfres.Cn
        #     # if self.sys.cdo != 0.0:
        #     #     Ccur[0, 0] += self.pdres.Cl
        #     #     Ccur[1, 0] += self.pdres.Cm
        #     #     Ccur[2, 0] += self.pdres.Cn
        # else:
        #     Ccur = zeros((0, 1), dtype=float)
        return Ccur

    def current_Dmat(self):
        numv = len(self.initstate)
        numc = len(self.initctrls)
        num = numv + numc
        Dcur = zeros((num, 1), dtype=float)
        j = 0
        for var in self.initstate:
            if var == 'alpha' or var == 'beta':
                Dcur[j, 0] = radians(getattr(self, var))
            else:
                Dcur[j, 0] = getattr(self, var)
            j += 1
        if self.trmmom:
            for ctrl in self.initctrls:
                Dcur[j, 0] = radians(self.ctrls[ctrl])
                j += 1

        # if self.trmmom:
        #     numc = len(self.sys.ctrls)
        # else:
        #     numc = 0
        # if self.trmlft:
        #     Dcur = zeros((1, 1), dtype=float)
        #     Dcur[0, 0] = radians(self.alpha)
        # else:
        #     num = numc+2
        #     Dcur = zeros((num, 1), dtype=float)
        #     Dcur[0, 0] = radians(self.alpha)
        #     Dcur[1, 0] = radians(self.beta)
        # if self.trmmom:
        #     c = 0
        #     for control in self.ctrls:
        #         Dcur[2+c, 0] = radians(self.ctrls[control])
        #         c += 1
        return Dcur

    def Hmat(self):
        numv = len(self.initstate)
        numc = len(self.initctrls)
        num = numv + numc
        numtgt = len(self.tgtlst)
        H = zeros((numtgt, num), dtype=float)
        for i, tgt in enumerate(self.tgtlst):
            j = 0
            for var in self.initstate:
                H[i, j] = getattr(getattr(self.stres, var), tgt)
                j += 1
            for ctrl in self.initctrls:
                control = self.ctrls[ctrl]
                if control < 0.0:
                    ctcp = self.gctrln_single(ctrl)
                else:
                    ctcp = self.gctrlp_single(ctrl)
                ctres = NearFieldResult(self, ctcp)
                H[i, j] = getattr(ctres, tgt)
                j += 1

        # if self.trmlft:
        #     H = zeros((1, 1), dtype=float)
        #     H[0, 0] = self.stres.alpha.CL
        # elif self.trmfrc and self.trmmom:
        #     H = zeros((5, num), dtype=float)
        #     H[0, 0] = self.stres.alpha.CL
        #     H[1, 0] = self.stres.alpha.CY
        #     H[2, 0] = self.stres.alpha.Cl
        #     H[3, 0] = self.stres.alpha.Cm
        #     H[4, 0] = self.stres.alpha.Cn
        #     H[0, 1] = self.stres.beta.CL
        #     H[1, 1] = self.stres.beta.CY
        #     H[2, 1] = self.stres.beta.Cl
        #     H[3, 1] = self.stres.beta.Cm
        #     H[4, 1] = self.stres.beta.Cn
        #     c = 0
        #     for control in self.ctrls:
        #         if self.ctrls[control] < 0.0:
        #             ctcp = self.gctrln_single(control)
        #         else:
        #             ctcp = self.gctrlp_single(control)
        #         ctres = NearFieldResult(self, ctcp)
        #         H[0, 2+c] = ctres.CL
        #         H[1, 2+c] = ctres.CY
        #         H[2, 2+c] = ctres.Cl
        #         H[3, 2+c] = ctres.Cm
        #         H[4, 2+c] = ctres.Cn
        #         c += 1
        # elif self.trmfrc:
        #     H = zeros((2, num), dtype=float)
        #     H[0, 0] = self.stres.alpha.CL
        #     H[1, 0] = self.stres.alpha.CY
        #     H[0, 1] = self.stres.beta.CL
        #     H[1, 1] = self.stres.beta.CY
        # elif self.trmmom:
        #     H = zeros((3, num), dtype=float)
        #     H[0, 0] = self.stres.alpha.Cl
        #     H[1, 0] = self.stres.alpha.Cm
        #     H[2, 0] = self.stres.alpha.Cn
        #     H[0, 1] = self.stres.beta.Cl
        #     H[1, 1] = self.stres.beta.Cm
        #     H[2, 1] = self.stres.beta.Cn
        #     ctgam = {}
        #     c = 0
        #     for control in self.ctrls:
        #         if self.ctrls[control] < 0.0:
        #             ctgam = self.gctrln_single(control)
        #         else:
        #             ctgam = self.gctrlp_single(control)
        #         ctres = NearFieldResult(self, ctgam)
        #         H[0, 2+c] = ctres.Cl
        #         H[1, 2+c] = ctres.Cm
        #         H[2, 2+c] = ctres.Cn
        #         c += 1
        # else:
        #     H = zeros((0, 0), dtype=float)
        return H

    def trim_iteration(self, display=False):
        # display = True
        Ctgt = self.target_Cmat()
        Ccur = self.current_Cmat()
        Cdff = Ctgt-Ccur
        if display:
            print(f'Cdff = \n{Cdff}\n')
        H = self.Hmat()
        if display:
            print(f'H = \n{H}\n')
        if H.shape[0] != H.shape[1]:
            A = H.transpose()*H
            B = H.transpose()*Cdff
        else:
            A = H
            B = Cdff
        if display:
            print(f'A = \n{A}\n')
            print(f'B = \n{B}\n')
        Ainv = inv(A)
        Dcur = self.current_Dmat()
        if display:
            print(f'Dcur = \n{Dcur}\n')
        Ddff = Ainv*B
        if display:
            print(f'Ddff = \n{Ddff}\n')
        Dcur = Dcur + Ddff
        return Dcur

    def trim(self, crit: float=1e-6, imax: int=100, display=False):
        # display = True
        Ctgt = self.target_Cmat()
        Ccur = self.current_Cmat()
        Cdff = Ctgt - Ccur
        nrmC = norm(Cdff)
        if display:
            print(f'normC = {nrmC}\n')
        iter = 0
        while nrmC > crit:
            if display:
                print(f'Iteration {iter:d}\n')
                start = perf_counter()
            self.reset()
            Dcur = self.trim_iteration(display=display)
            if display:
                finish = perf_counter()
                elapsed = finish-start
                print(f'Trim Internal Iteration Duration = {elapsed:.3f} seconds.\n')
            if Dcur is False:
                return
            j = 0
            for var in self.initstate:
                curstate = {}
                if var == 'alpha' or var == 'beta':
                    curstate[var] = degrees(Dcur[j, 0])
                else:
                    curstate[var] = Dcur[j, 0]
                j += 1
                self.set_state(**curstate)
            if self.trmmom:
                curctrls = {}
                for ctrl in self.initctrls:
                    curctrls[ctrl] = degrees(Dcur[j, 0])
                    j += 1
                self.set_controls(**curctrls)
            Ccur = self.current_Cmat()
            Cdff = Ctgt-Ccur
            nrmC = norm(Cdff)
            if display:
                print(f'Ctgt = \n{Ctgt}\n')
                print(f'Ccur = \n{Ccur}\n')
                print(f'Cdff = \n{Cdff}\n')
                print(f'alpha = {self.alpha:.6f} deg')
                print(f'beta = {self.beta:.6f} deg')
                print(f'pbo2V = {self.pbo2V}')
                print(f'qco2V = {self.qco2V}')
                print(f'rbo2V = {self.rbo2V}')
                for ctrl in self.ctrls:
                    print(f'{ctrl} = {self.ctrls[ctrl]:.6f} deg')
                print(f'normC = {nrmC}\n')

            check = False
            if abs(self.alpha) > ANGTOL:
                check = True
            elif abs(self.beta) > ANGTOL:
                check = True
            else:
                 for ctrl in self.ctrls:
                     if abs(self.ctrls[ctrl]) > ANGTOL:
                         check = True
                         break

            iter += 1
            if iter >= imax or check:
                print(f'Ctgt = \n{Ctgt}\n')
                print(f'Ccur = \n{Ccur}\n')
                print(f'Cdff = \n{Cdff}\n')
                print(f'alpha = {self.alpha:.6f} deg')
                print(f'beta = {self.beta:.6f} deg')
                print(f'pbo2V = {self.pbo2V}')
                print(f'qco2V = {self.qco2V}')
                print(f'rbo2V = {self.rbo2V}')
                for ctrl in self.ctrls:
                    print(f'{ctrl} = {self.ctrls[ctrl]:.6f} deg')
                print(f'Convergence failed for {self.name:s}.')
                return False
        print(f'Converged {self.name:s} in {iter:d} iterations.')

def paneltrim_from_dict(psys: 'PanelSystem', resdata: dict):
    name = resdata['name']

    if resdata['trim'] == 'Load Trim':
        trim = LoadTrim(name, psys)
        L, Y, l, m, n = None, None, None, None, None
        if 'L' in resdata:
            L = resdata['L']
        if 'Y' in resdata:
            Y = resdata['Y']
        if 'l' in resdata:
            l = resdata['l']
        if 'm' in resdata:
            m = resdata['m']
        if 'n' in resdata:
            n = resdata['n']
        trim.set_loads(L, Y, l, m, n)
    elif resdata['trim'] == 'Looping Trim':
        trim = LoopingTrim(name, psys)
        lf = 1.0
        if 'load factor' in resdata:
            lf = resdata['load factor']
        trim.set_load_factor(lf)
    elif resdata['trim'] == 'Turning Trim':
        trim = TurningTrim(name, psys)
        bang = 0.0
        if 'bank angle' in resdata:
            bang = resdata['bank angle']
        trim.set_bank_angle(bang)

    rho = 1.0
    if 'density' in resdata:
        rho = resdata['density']
    speed = 1.0
    if 'speed' in resdata:
        speed = resdata['speed']
    trim.set_speed_and_density(speed, rho)

    initstate = {}
    if 'alpha' in resdata:
        initstate['alpha'] = resdata['alpha']
    if 'beta' in resdata:
        initstate['beta'] = resdata['beta']
    if 'pbo2V' in resdata:
        initstate['pbo2V'] = resdata['pbo2V']
    if 'qco2V' in resdata:
        initstate['qco2V'] = resdata['qco2V']
    if 'rbo2V' in resdata:
        initstate['rbo2V'] = resdata['rbo2V']
    trim.set_initial_state(initstate)

    initctrls = {}
    for ctrl in psys.ctrls:
        if ctrl in resdata:
            initctrls[ctrl] = resdata[ctrl]
    trim.set_initial_controls(initctrls)

    if isinstance(trim, (LoopingTrim, TurningTrim)):

        if 'gravacc' in resdata:
            g = resdata['gravacc']
            trim.set_gravitational_acceleration(g)

        mval = 1.0
        xcm, ycm, zcm = psys.rref.x, psys.rref.y, psys.rref.z
        if 'mass' in resdata:
            if isinstance(resdata['mass'], str):
                mass = psys.masses[resdata['mass']]
            elif isinstance(resdata['mass'], float):
                if 'rcg' in resdata:
                    rcgdata = resdata['rcg']
                    xcm, ycm, zcm = rcgdata['x'], rcgdata['y'], rcgdata['z']
                mval = resdata['mass']
                mass = Mass(name + ' Mass', mval, xcm, ycm, zcm)
        else:
            mass = Mass(name + ' Mass', mval, xcm, ycm, zcm)

        trim.set_mass(mass)

    ptrm = trim.create_trim_result()

    if 'mach' in resdata:
        mach = resdata['mach']
        ptrm.set_state(mach=mach)

    # trim_force = True
    # trim_moment = True
    # trim_lift = False
    # if 'trim moment' in resdata:
    #     trim_moment = resdata['trim moment']
    # if 'trim lift' in resdata:
    #     trim_lift = resdata['trim lift']
    # if trim_lift:
    #     trim_force = False
    #     trim_moment = False

    # ptrm.set_trim_loads(trmfrc=trim_force, trmmom=trim_moment, trmlft=trim_lift)

    psys.results[name] = ptrm

    ptrm.trim()

    return ptrm
