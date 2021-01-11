from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.matrix3d import elementwise_dot_product, elementwise_multiply, elementwise_cross_product
from numpy.matlib import matrix, ones, zeros
from math import cos, sin, radians
from numpy.matlib import sqrt, square, multiply, absolute
from matplotlib.pyplot import figure

tol = 1e-12

class PanelResult(object):
    name: str = None
    sys = None
    rho: float = None
    mach: float = None
    speed: float = None
    alpha: float = None
    beta: float = None
    pbo2V: float = None
    qco2V: float = None
    rbo2V: float = None
    _acs: Coordinate = None
    _wcs: Coordinate = None
    _vfs: Vector = None
    _ofs: Vector = None
    _qfs: float = None
    _unsig: matrix = None
    _unmu: matrix = None
    _sig: matrix = None
    _mu: matrix = None
    _nfres = None
    _strpres = None
    _ffres = None
    _stres = None
    _vfsl: MatrixVector = None
    def __init__(self, name: str, sys):
        self.name = name
        self.sys = sys
        self.initialise()
    def initialise(self):
        self.rho = 1.0
        self.mach = 0.0
        self.speed = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        self.pbo2V = 0.0
        self.qco2V = 0.0
        self.rbo2V = 0.0
        self.rcg = self.sys.rref
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    def set_density(self, rho: float=None):
        if rho is not None:
            self.rho = rho
        self.reset()
    def set_state(self, mach: float=None, speed: float=None,
                  alpha: float=None, beta: float=None,
                  pbo2V: float=None, qco2V: float=None, rbo2V: float=None):
        if mach is not None:
            self.mach = mach
        if speed is not None:
            self.speed = speed
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if pbo2V is not None:
            self.pbo2V = pbo2V
        if qco2V is not None:
            self.qco2V = qco2V
        if rbo2V is not None:
            self.rbo2V = rbo2V
        self.reset()
    def set_cg(self, rcg: Vector):
        self.rcg = rcg
        self.reset()
    @property
    def acs(self):
        if self._acs is None:
            pnt = self.sys.rref
            cosal, sinal = trig_angle(self.alpha)
            cosbt, sinbt = trig_angle(self.beta)
            dirx = Vector(cosbt*cosal, -sinbt, cosbt*sinal)
            diry = Vector(sinbt*cosal, cosbt, sinbt*sinal)
            dirz = Vector(-sinal, 0.0, cosal)
            self._acs = Coordinate(pnt, dirx, diry, dirz)
        return self._acs
    @property
    def wcs(self):
        if self._wcs is None:    
            pnt = self.sys.rref
            dirx = -1.0*self.acs.dirx
            diry = self.acs.diry
            dirz = -1.0*self.acs.dirz
            self._wcs = Coordinate(pnt, dirx, diry, dirz)
        return self._wcs
    @property
    def vfs(self):
        if self._vfs is None:
            if self.alpha is None:
                self.alpha = 0.0
            if self.beta is None:
                self.beta = 0.0
            if self.speed is None:
                self.speed = 1.0
            self._vfs = self.acs.dirx*self.speed
        return self._vfs
    def calc_ofs(self, pbo2V: float, qco2V: float, rbo2V: float):
        p = pbo2V*2*self.speed/self.sys.bref
        q = qco2V*2*self.speed/self.sys.cref
        r = rbo2V*2*self.speed/self.sys.bref
        rotvl = Vector(p, q, r)
        return self.wcs.vector_to_global(rotvl)
    @property
    def ofs(self):
        if self._ofs is None:
            self._ofs = self.calc_ofs(self.pbo2V, self.qco2V, self.rbo2V)
        return self._ofs
    @property
    def vfsl(self):
        if self._vfsl is None:
            self._vfsl = zero_matrix_vector((self.sys.numpnl, 1), dtype=float)
            for pnl in self.sys.pnls.values():
                self._vfsl[pnl.ind, 0] = pnl.crd.vector_to_local(self.vfs)
        return self._vfsl
    @property
    def qfs(self):
        if self._qfs is None:
            self._qfs = self.rho*self.speed**2/2
        return self._qfs
    @property
    def unsig(self):
        if self._unsig is None:
            self._unsig = self.sys.sig # ungam(self.mach)
        return self._unsig
    @property
    def unmu(self):
        if self._unmu is None:
            self._unmu = self.sys.mu # ungam(self.mach)
        return self._unmu
    @property
    def sig(self):
        if self._sig is None:
            self._sig = self.unsig[:, 0]*self.vfs + self.unsig[:, 1]*self.ofs
        return self._sig
    @property
    def mu(self):
        if self._mu is None:
            self._mu = self.unmu[:, 0]*self.vfs + self.unmu[:, 1]*self.ofs
        return self._mu
    @property
    def nfres(self):
        if self._nfres is None:
            self._nfres = NearFieldResult(self)
        return self._nfres
    @property
    def strpres(self):
        if self._strpres is None:
            if self.sys.srfcs is not None:
                self._strpres = StripResult(self.nfres)
        return self._strpres
    @property
    def ffres(self):
        if self._ffres is None:
            if self.sys.srfcs is not None:
                self._ffres = FarFieldResult(self)
        return self._ffres            
    @property
    def stres(self):
        if self._stres is None:
            self._stres = StabilityResult(self)
        return self._stres
    def plot_strip_lift_force_distribution(self, ax=None, axis: str='b',
                                           surfaces: list=None, normalise: bool=False):
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
            for srfc in srfcs:
                if normalise:
                    l = [self.strpres.lift[strp.ind, 0]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    l = [self.strpres.lift[strp.ind, 0]/strp.width for strp in srfc.strps]
                label = self.name+' for '+srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, l, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, l, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(l, z, label=label)
            ax.legend()
        return ax
    def plot_strip_side_force_distribution(self, ax=None, axis: str='b',
                                           surfaces: list=None, normalise: bool=False):
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
            for srfc in srfcs:
                if normalise:
                    f = [self.strpres.side[strp.ind, 0]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    f = [self.strpres.side[strp.ind, 0]/strp.width for strp in srfc.strps]
                label = self.name+' for '+srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, f, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, f, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(f, z, label=label)
            ax.legend()
        return ax
    def plot_strip_drag_force_distribution(self, ax=None, axis: str='b',
                                           surfaces: list=None, normalise: bool=False):
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
            for srfc in srfcs:
                if normalise:
                    d = [self.strpres.drag[strp.ind, 0]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    d = [self.strpres.drag[strp.ind, 0]/strp.width for strp in srfc.strps]
                label = self.name + ' for ' + srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, d, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, d, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(d, z, label=label)
            ax.legend()
        return ax
    def plot_trefftz_lift_force_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False):
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
            for srfc in srfcs:
                if normalise:
                    l = [self.ffres.lift[strp.ind, 0]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    l = [self.ffres.lift[strp.ind, 0]/strp.width for strp in srfc.strps]
                label = self.name+' for '+srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, l, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, l, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(l, z, label=label)
            ax.legend()
        return ax
    def plot_trefftz_side_force_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False):
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
            for srfc in srfcs:
                if normalise:
                    f = [self.ffres.side[strp.ind, 0]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    f = [self.ffres.side[strp.ind, 0]/strp.width for strp in srfc.strps]
                label = self.name+' for '+srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, f, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, f, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(f, z, label=label)
            ax.legend()
        return ax
    def plot_trefftz_drag_force_distribution(self, ax=None, axis: str='b',
                                             surfaces: list=None, normalise: bool=False):
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
            for srfc in srfcs:
                if normalise:
                    d = [self.ffres.drag[strp.ind, 0]/strp.area/self.qfs for strp in srfc.strps]
                else:
                    d = [self.ffres.drag[strp.ind, 0]/strp.width for strp in srfc.strps]
                label = self.name + ' for ' + srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, d, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, d, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(d, z, label=label)
            ax.legend()
        return ax
    def plot_trefftz_down_wash_distribution(self, ax=None, axis: str='b',
                                            surfaces: list=None):
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
            for srfc in srfcs:
                w = [self.ffres.wash[strp.ind, 0] for strp in srfc.strps]
                # wa = [self.ffres.washa[strp.ind, 0] for strp in srfc.strps]
                # wb = [self.ffres.washb[strp.ind, 0] for strp in srfc.strps]
                label = self.name + ' for ' + srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, w, label=label)
                        # ax.plot(b, wa, label=label)
                        # ax.plot(b, wb, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, w, label=label)
                        # ax.plot(y, wa, label=label)
                        # ax.plot(y, wb, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(w, z, label=label)
                        # ax.plot(wa, z, label=label)
                        # ax.plot(wb, z, label=label)
            ax.legend()
        return ax
    def plot_trefftz_circulation_distribution(self, ax=None, axis: str='b',
                                              surfaces: list=None):
        if self.sys.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
                srfcs = [srfc for srfc in self.sys.srfcs]
            else:
                srfcs = []
                for srfc in self.sys.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            for srfc in srfcs:
                c = [self.ffres.circ[strp.ind, 0] for strp in srfc.strps]
                # ma = [self.ffres.mua[strp.ind, 0] for strp in srfc.strps]
                # mb = [self.ffres.mub[strp.ind, 0] for strp in srfc.strps]
                label = self.name + ' for ' + srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, c, label=label)
                        # ax.plot(b, ma, label=label)
                        # ax.plot(b, mb, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, c, label=label)
                        # ax.plot(y, ma, label=label)
                        # ax.plot(y, mb, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(c, z, label=label)
                        # ax.plot(ma, z, label=label)
                        # ax.plot(mb, z, label=label)
            ax.legend()
        return ax
    @property
    def stability_derivatives(self):
        return self.stres.stability_derivatives
    @property
    def stability_derivatives_body(self):
        return self.stres.stability_derivatives_body
    @property
    def surface_loads(self):
        if self.sys.srfcs is not None:
            from py2md.classes import MDTable, MDReport, MDHeading
            from math import atan
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
                frc = self.nfres.nffrc[ind, 0].sum()
                mom = self.nfres.nfmom[ind, 0].sum()
                table1.add_row([srfc.name, frc.x, frc.y, frc.z, mom.x, mom.y, mom.z])
                if area > 0.0:
                    Di = frc*self.acs.dirx
                    Y = frc*self.acs.diry
                    L = frc*self.acs.dirz
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
            table2.add_row(['Total', self.sys.sref, Ditot, Ytot, Ltot, self.nfres.CDi, self.nfres.CY, self.nfres.CL])
            report.add_object(table2)
            return report
    def __str__(self):
        from py2md.classes import MDTable
        from . import cfrm, dfrm, efrm
        outstr = '# Panel Result '+self.name+' for '+self.sys.name+'\n'
        table = MDTable()
        table.add_column('Alpha (deg)', cfrm, data=[self.alpha])
        table.add_column('Beta (deg)', cfrm, data=[self.beta])
        table.add_column('Speed', cfrm, data=[self.speed])
        table.add_column('Rho', cfrm, data=[self.rho])
        table.add_column('Mach', efrm, data=[self.mach])
        outstr += table._repr_markdown_()
        table = MDTable()
        table.add_column('pb/2V (rad)', cfrm, data=[self.pbo2V])
        table.add_column('qc/2V (rad)', cfrm, data=[self.qco2V])
        table.add_column('rb/2V (rad)', cfrm, data=[self.rbo2V])
        outstr += table._repr_markdown_()
        table = MDTable()
        table.add_column('xcg', '.5f', data=[self.rcg.x])
        table.add_column('ycg', '.5f', data=[self.rcg.y])
        table.add_column('zcg', '.5f', data=[self.rcg.z])
        outstr += table._repr_markdown_()
        # if len(self.ctrls) > 0:
        #     table = MDTable()
        #     for control in self.ctrls:
        #         ctrl = self.ctrls[control]
        #         control = control.capitalize()
        #         table.add_column(f'{control} (deg)', cfrm, data=[ctrl])
        #     outstr += table._repr_markdown_()
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
            outstr += table._repr_markdown_()
            table = MDTable()
            table.add_column('CDi', dfrm, data=[self.nfres.CDi])
            table.add_column('CY', cfrm, data=[self.nfres.CY])
            table.add_column('CL', cfrm, data=[self.nfres.CL])
            table.add_column('Cl', cfrm, data=[self.nfres.Cl])
            table.add_column('Cm', cfrm, data=[self.nfres.Cm])
            table.add_column('Cn', cfrm, data=[self.nfres.Cn])
            table.add_column('e', efrm, data=[self.nfres.e])
            # if self.sys.cdo != 0.0:
            #     lod = self.nfres.CL/(self.pdres.CDo+self.nfres.CDi)
            #     table.add_column('L/D', '.5g', data=[lod])
            outstr += table._repr_markdown_()
        # if self.phi is not None:
        #     table = MDTable()
        #     table.add_column('CDi_ff', dfrm, data=[self.trres.CDi])
        #     table.add_column('CY_ff', cfrm, data=[self.trres.CY])
        #     table.add_column('CL_ff', cfrm, data=[self.trres.CL])
        #     # table.add_column('Cl_ff', cfrm, data=[self.trres.Cl])
        #     # table.add_column('Cm_ff', cfrm, data=[self.trres.Cm])
        #     # table.add_column('Cn_ff', cfrm, data=[self.trres.Cn])
        #     table.add_column('e', efrm, data=[self.trres.e])
        #     if self.sys.cdo != 0.0:
        #         lod_ff = self.trres.CL/(self.pdres.CDo+self.trres.CDi)
        #         table.add_column('L/D_ff', '.5g', data=[lod_ff])
        #     outstr += table._repr_markdown_()
        return outstr
    def __repr__(self):
        return f'<PanelResult: {self.name}>'
    def _repr_markdown_(self):
        return self.__str__()

def trig_angle(angle: float):
    '''Calculates cos(angle) and sin(angle) with angle in degrees.'''
    angrad = radians(angle)
    cosang = cos(angrad)
    sinang = sin(angrad)
    return cosang, sinang

class NearFieldResult(object):
    res = None
    sig: matrix = None
    mu: matrix = None
    _nfql = None
    _nfqt = None
    _nfcp = None
    _nfphi = None
    _nfprs = None
    _nffrc = None
    _nfmom = None
    _nffrctot = None
    _nfmomtot = None
    _Cx = None
    _Cy = None
    _Cz = None
    _Cmx = None
    _Cmy = None
    _Cmz = None
    _CDi = None
    _CY = None
    _CL = None
    _e = None
    _Cl = None
    _Cm = None
    _Cn = None
    def __init__(self, res: PanelResult, sig: matrix=None, mu: matrix=None):
        self.res = res
        if sig is None:
            self.sig = self.res.sig
        else:
            self.sig = sig
        if mu is None:
            self.mu = self.res.mu
        else:
            self.mu = mu
    @property
    def nfql(self):
        if self._nfql is None:
            self._nfql = zero_matrix_vector((self.res.sys.numpnl, 1), dtype=float)
            for pnl in self.res.sys.pnls.values():
                qx, qy = pnl.diff_mu(self.mu)
                vfsl = self.res.vfsl[pnl.ind, 0]
                self._nfql[pnl.ind, 0] = Vector(qx + vfsl.x, qy + vfsl.y, 0.0)
        return self._nfql
    @property
    def nfqt(self):
        if self._nfqt is None:
            self._nfqt = self.nfql.return_magnitude()
        return self._nfqt
    @property
    def nfcp(self):
        if self._nfcp is None:
            self._nfcp = 1.0 - square(self.nfqt)/self.res.speed**2
        return self._nfcp
    @property
    def nfphi(self):
        if self._nfphi is None:
            self._nfphi = self.res.sys.apm*self.mu + self.res.sys.aps*self.sig
            self._nfphi[absolute(self._nfphi) < tol] = 0.0
        return self._nfphi
    @property
    def nfprs(self):
        if self._nfprs is None:
            self._nfprs = self.res.qfs*self.nfcp
        return self._nfprs
    @property
    def nffrc(self):
        if self._nffrc is None:
            self._nffrc = -elementwise_multiply(self.res.sys.nrms, multiply(self.nfprs, self.res.sys.pnla))
        return self._nffrc
    @property
    def nfmom(self):
        if self._nfmom is None:
            self._nfmom = elementwise_cross_product(self.res.sys.rrel, self.nffrc)
        return self._nfmom
    @property
    def nffrctot(self):
        if self._nffrctot is None:
            self._nffrctot = self.nffrc.sum()
        return self._nffrctot
    @property
    def nfmomtot(self):
        if self._nfmomtot is None:
            self._nfmomtot = self.nfmom.sum()
        return self._nfmomtot
    @property
    def Cx(self):
        if self._Cx is None:
            self._Cx = self.nffrctot.x/self.res.qfs/self.res.sys.sref
            self._Cx = fix_zero(self._Cx)
        return self._Cx
    @property
    def Cy(self):
        if self._Cy is None:
            self._Cy = self.nffrctot.y/self.res.qfs/self.res.sys.sref
            self._Cy = fix_zero(self._Cy)
        return self._Cy
    @property
    def Cz(self):
        if self._Cz is None:
            self._Cz = self.nffrctot.z/self.res.qfs/self.res.sys.sref
            self._Cz = fix_zero(self._Cz)
        return self._Cz
    @property
    def Cmx(self):
        if self._Cmx is None:
            self._Cmx = self.nfmomtot.x/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cmx = fix_zero(self._Cmx)
        return self._Cmx
    @property
    def Cmy(self):
        if self._Cmy is None:
            self._Cmy = self.nfmomtot.y/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cmy = fix_zero(self._Cmy)
        return self._Cmy
    @property
    def Cmz(self):
        if self._Cmz is None:
            self._Cmz = self.nfmomtot.z/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cmz = fix_zero(self._Cmz)
        return self._Cmz
    @property
    def CDi(self):
        if self._CDi is None:
            Di = self.res.acs.dirx*self.nffrctot
            self._CDi = Di/self.res.qfs/self.res.sys.sref
            self._CDi = fix_zero(self._CDi)
        return self._CDi
    @property
    def CY(self):
        if self._CY is None:
            Y = self.res.acs.diry*self.nffrctot
            self._CY = Y/self.res.qfs/self.res.sys.sref
            self._CY = fix_zero(self._CY)
        return self._CY
    @property
    def CL(self):
        if self._CL is None:
            L = self.res.acs.dirz*self.nffrctot
            self._CL = L/self.res.qfs/self.res.sys.sref
            self._CL = fix_zero(self._CL)
        return self._CL
    @property
    def Cl(self):
        if self._Cl is None:
            l = self.res.wcs.dirx*self.nfmomtot
            self._Cl = l/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cl = fix_zero(self._Cl)
        return self._Cl
    @property
    def Cm(self):
        if self._Cm is None:
            m = self.res.wcs.diry*self.nfmomtot
            self._Cm = m/self.res.qfs/self.res.sys.sref/self.res.sys.cref
            self._Cm = fix_zero(self._Cm)
        return self._Cm
    @property
    def Cn(self):
        if self._Cn is None:
            n = self.res.wcs.dirz*self.nfmomtot
            self._Cn = n/self.res.qfs/self.res.sys.sref/self.res.sys.bref
            self._Cn = fix_zero(self._Cn)
        return self._Cn
    @property
    def e(self):
        if self._e is None:
            if self.CDi <= 0.0:
                self._e = float('nan')
            elif self.CL == 0.0 and self.CY == 0.0:
                self._e = 0.0
            else:
                from math import pi
                self._e = (self.CL**2+self.CY**2)/pi/self.res.sys.ar/self.CDi
                self._e = fix_zero(self._e)
        return self._e

class StripResult(object):
    nfres = None
    _stfrc = None
    _lift = None
    _side = None
    _drag = None
    def __init__(self, nfres: NearFieldResult):
        self.nfres = nfres
    @property
    def stfrc(self):
        if self._stfrc is None:
            sys = self.nfres.res.sys
            num = len(sys.strps)
            self._stfrc = zero_matrix_vector((num, 1))
            for strp in sys.strps:
                i = strp.ind
                for pnl in strp.pnls:
                    j = pnl.ind
                    self._stfrc[i, 0] = self._stfrc[i, 0]+self.nfres.nffrc[j, 0]
        return self._stfrc
    @property
    def drag(self):
        if self._drag is None:
            self._drag = zeros(self.stfrc.shape, dtype=float)
            for i in range(self.stfrc.shape[0]):
                self._drag[i, 0] = self.nfres.res.acs.dirx*self.stfrc[i, 0]
        return self._drag
    @property
    def side(self):
        if self._side is None:
            self._side = zeros(self.stfrc.shape, dtype=float)
            for i in range(self.stfrc.shape[0]):
                self._side[i, 0] = self.nfres.res.acs.diry*self.stfrc[i, 0]
        return self._side
    @property
    def lift(self):
        if self._lift is None:
            self._lift = zeros(self.stfrc.shape, dtype=float)
            for i in range(self.stfrc.shape[0]):
                self._lift[i, 0] = self.nfres.res.acs.dirz*self.stfrc[i, 0]
        return self._lift

class FarFieldResult(object):
    res = None
    _ffmu = None
    _ffwsh = None
    _fffrc = None
    _ffmom = None
    _fffrctot = None
    _ffmomtot = None
    # _mua = None
    # _mub = None
    _circ = None
    # _washa = None
    # _washb = None
    _wash = None
    _wash_v2 = None
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
            self._ffmu = zeros((numhsv, 1), dtype=float)
            for ind, hsv in enumerate(self.res.sys.hsvs):
                self._ffmu[ind, 0] = self.res.mu[hsv.ind, 0]
        return self._ffmu
    @property
    def ffwsh(self):
        if self._ffwsh is None:
            self._ffwsh = self.res.sys.awh*self.ffmu
            # self._ffwsh += self.res.sys.hsvnrms*self.res.vfs
            # self._ffwsh += self.res.sys.awd*self.res.mu
            # self._ffwsh += self.res.sys.aws*self.res.sig
        return self._ffwsh
    @property
    def fffrc(self):
        if self._fffrc is None:
            from pygeom.matrix3d import MatrixVector
            x = self.res.rho*multiply(self.ffmu, self.res.sys.adh*self.ffmu)
            y = self.res.rho*self.res.speed*multiply(self.ffmu, self.res.sys.ash)
            z = self.res.rho*self.res.speed*multiply(self.ffmu, self.res.sys.alh)
            self._fffrc = MatrixVector(x, y, z)
        return self._fffrc
    @property
    def ffmom(self):
        if self._ffmom is None:
            self._ffmom = elementwise_cross_product(self.res.brm, self.fffrc)
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
    # @property
    # def mua(self):
    #     if self._mua is None:
    #         sys = self.res.sys
    #         num = len(sys.strps)
    #         self._mua = zeros((num, 1), dtype=float)
    #         for i, strp in enumerate(sys.strps):
    #             pnla = strp.pnls[0]
    #             self._mua[i, 0] = self.res.mu[pnla.ind]
    #     return self._mua
    # @property
    # def mub(self):
    #     if self._mub is None:
    #         sys = self.res.sys
    #         num = len(sys.strps)
    #         self._mub = zeros((num, 1), dtype=float)
    #         for i, strp in enumerate(sys.strps):
    #             pnlb = strp.pnls[-1]
    #             self._mub[i, 0] = self.res.mu[pnlb.ind]
    #     return self._mub
    @property
    def circ(self):
        if self._circ is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._circ = zeros((num, 1), dtype=float)
            for i, strp in enumerate(sys.strps):
                pnla = strp.pnls[0]
                pnlb = strp.pnls[-1]
                mua = self.res.mu[pnla.ind]
                mub = self.res.mu[pnlb.ind]
                self._circ[i, 0] = mub-mua
        return self._circ
    # @property
    # def washa(self):
    #     if self._washa is None:
    #         sys = self.res.sys
    #         num = len(sys.strps)
    #         self._washa = zeros((num, 1), dtype=float)
    #         for i, strp in enumerate(sys.strps):
    #             pnl = strp.pnls[0]
    #             pind = pnl.ind
    #             hinds = sys.phind[pind]
    #             self._washa[i, 0] += self.ffwsh[hinds[0], 0]
    #     return self._washa
    # @property
    # def washb(self):
    #     if self._washb is None:
    #         sys = self.res.sys
    #         num = len(sys.strps)
    #         self._washb = zeros((num, 1), dtype=float)
    #         for i, strp in enumerate(sys.strps):
    #             pnl = strp.pnls[-1]
    #             pind = pnl.ind
    #             hinds = sys.phind[pind]
    #             self._washb[i, 0] = self.ffwsh[hinds[0], 0]
    #     return self._washb
    @property
    def wash_v2(self):
        if self._wash_v2 is None:
            sys = self.res.sys
            num = len(sys.strps)
            index = []
            for strp in sys.strps:
                hinds = sys.phind[strp.pnls[0].ind]
                index.append(hinds[0])
            num = len(index)
            awh = zeros((num, num), dtype=float)
            for i, indi in enumerate(index):
                for j, indj in enumerate(index):
                    awh[i, j] = sys.awh[indi, indj]
            # awh = self.res.sys.awh[index, index]
            self._wash_v2 = awh*self.circ
        return self._wash_v2
    @property
    def wash(self):
        if self._wash is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._wash = zeros((num, 1), dtype=float)
            for i, strp in enumerate(sys.strps):
                pnla = strp.pnls[0]
                pnlb = strp.pnls[-1]
                # dista = (pnla.edgpnts[3]-pnla.pnto).return_magnitude()
                # distb = (pnlb.edgpnts[1]-pnla.pnto).return_magnitude()
                # dist = (pnlb.pnto - pnla.pnto).return_magnitude()
                # dist = dista+distb
                # mua = self.res.mu[pnla.ind]
                # mub = self.res.mu[pnlb.ind]
                # self._wash[i, 0] = (mub-mua)/dist
                pinda = pnla.ind
                pindb = pnlb.ind
                hindsa = sys.phind[pinda]
                hindsb = sys.phind[pindb]
                cnt = 0
                for hind in hindsa:
                    cnt += 1
                    self._wash[i, 0] -= self.ffwsh[hind, 0]
                for hind in hindsb:
                    cnt += 1
                    self._wash[i, 0] += self.ffwsh[hind, 0]
                self._wash[i, 0] = self._wash[i, 0]/cnt/2
        return self._wash
    @property
    def drag(self):
        if self._drag is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._drag = zeros((num, 1), dtype=float)
            for i, strp in enumerate(sys.strps):
                self._drag[i, 0] = self.res.rho*self.wash_v2[i, 0]*self.circ[i, 0]*strp.width
                # pnla = strp.pnls[0]
                # pnlb = strp.pnls[-1]
                # pinda = pnla.ind
                # pindb = pnlb.ind
                # hindsa = sys.phind[pinda]
                # hindsb = sys.phind[pindb]
                # for hind in hindsa:
                #     self._drag[i, 0] += self.fffrc[hind, 0].x
                # for hind in hindsb:
                #     self._drag[i, 0] += self.fffrc[hind, 0].x
        return self._drag
    @property
    def side(self):
        if self._side is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._side = zeros((num, 1), dtype=float)
            for i, strp in enumerate(sys.strps):
                pnla = strp.pnls[0]
                pnlb = strp.pnls[-1]
                pinda = pnla.ind
                pindb = pnlb.ind
                hindsa = sys.phind[pinda]
                hindsb = sys.phind[pindb]
                for hind in hindsa:
                    self._side[i, 0] += self.fffrc[hind, 0].y
                for hind in hindsb:
                    self._side[i, 0] += self.fffrc[hind, 0].y
        return self._side
    @property
    def lift(self):
        if self._lift is None:
            sys = self.res.sys
            num = len(sys.strps)
            self._lift = zeros((num, 1), dtype=float)
            for i, strp in enumerate(sys.strps):
                pnla = strp.pnls[0]
                pnlb = strp.pnls[-1]
                pinda = pnla.ind
                pindb = pnlb.ind
                hindsa = sys.phind[pinda]
                hindsb = sys.phind[pindb]
                for hind in hindsa:
                    self._lift[i, 0] += self.fffrc[hind, 0].z
                for hind in hindsb:
                    self._lift[i, 0] += self.fffrc[hind, 0].z
        return self._lift
    @property
    def CDi(self):
        if self._CDi is None:
            Di = self.fffrctot.x
            self._CDi = Di/self.res.qfs/self.res.sys.sref
            self._CDi = fix_zero(self._CDi)
        return self._CDi
    @property
    def CY(self):
        if self._CY is None:
            Y = self.fffrctot.y
            self._CY = Y/self.res.qfs/self.res.sys.sref
            self._CY = fix_zero(self._CY)
        return self._CY
    @property
    def CL(self):
        if self._CL is None:
            L = self.fffrctot.z
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
                from math import pi
                self._e = (self.CL**2+self.CY**2)/pi/self.res.sys.ar/self.CDi
                self._e = fix_zero(self._e)
        return self._e

class StabilityResult(object):
    res = None
    _u = None
    _v = None
    _w = None
    _p = None
    _q = None
    _r = None
    _alpha = None
    _beta = None
    _pbo2V = None
    _qco2V = None
    _rbo2V = None
    _pdbo2V = None
    _qdco2V = None
    _rdbo2V = None
    def __init__(self, res: PanelResult):
        self.res = res
    @property
    def u(self):
        if self._u is None:
            uvw = Vector(1.0, 0.0, 0.0)
            sigu = self.res.unsig[:, 0]*uvw
            muu = self.res.unmu[:, 0]*uvw
            self._u = NearFieldResult(self.res, sig=sigu, mu=muu)
        return self._u
    @property
    def v(self):
        if self._v is None:
            uvw = Vector(0.0, 1.0, 0.0)
            sigv = self.res.unsig[:, 0]*uvw
            muv = self.res.unmu[:, 0]*uvw
            self._v = NearFieldResult(self.res, sig=sigv, mu=muv)
        return self._v
    @property
    def w(self):
        if self._w is None:
            uvw = Vector(0.0, 0.0, 1.0)
            sigw = self.res.unsig[:, 0]*uvw
            muw = self.res.unmu[:, 0]*uvw
            self._w = NearFieldResult(self.res, sig=sigw, mu=muw)
        return self._w
    @property
    def p(self):
        if self._p is None:
            pqr = Vector(1.0, 0.0, 0.0)
            ofs = self.res.wcs.vector_to_global(pqr)
            sigp = self.res.unsig[:, 1]*ofs
            mup = self.res.unmu[:, 1]*ofs
            self._p = NearFieldResult(self.res, sig=sigp, mu=mup)
        return self._p
    @property
    def q(self):
        if self._q is None:
            pqr = Vector(0.0, 1.0, 0.0)
            ofs = self.res.wcs.vector_to_global(pqr)
            sigq = self.res.unsig[:, 1]*ofs
            muq = self.res.unmu[:, 1]*ofs
            self._q = NearFieldResult(self.res, sig=sigq, mu=muq)
        return self._q
    @property
    def r(self):
        if self._r is None:
            pqr = Vector(0.0, 0.0, 1.0)
            ofs = self.res.wcs.vector_to_global(pqr)
            sigr = self.res.unsig[:, 1]*ofs
            mur = self.res.unmu[:, 1]*ofs
            self._r = NearFieldResult(self.res, sig=sigr, mu=mur)
        return self._r
    @property
    def alpha(self):
        if self._alpha is None:
            V = self.res.speed
            c = self.res.sys.cref
            b = self.res.sys.bref
            pbo2V = self.res.pbo2V
            qco2V = self.res.qco2V
            rbo2V = self.res.rbo2V
            cosal, sinal = trig_angle(self.res.alpha)
            cosbt, sinbt = trig_angle(self.res.beta)
            vfs = Vector(-V*cosbt*sinal, 0, V*cosal*cosbt)
            ofs = Vector(2*V*(qco2V*sinal*sinbt/c - cosal*rbo2V/b - cosbt*pbo2V*sinal/b), 0.0,
                         2*V*(cosal*cosbt*pbo2V/b - cosal*qco2V*sinbt/c - rbo2V*sinal/b))
            sigalpha = self.res.unsig[:, 0]*vfs+self.res.unsig[:, 1]*ofs
            mualpha = self.res.unmu[:, 0]*vfs+self.res.unmu[:, 1]*ofs
            self._alpha = NearFieldResult(self.res, sig=sigalpha, mu=mualpha)
        return self._alpha
    @property
    def beta(self):
        if self._beta is None:
            V = self.res.speed
            c = self.res.sys.cref
            b = self.res.sys.bref
            pbo2V = self.res.pbo2V
            qco2V = self.res.qco2V
            cosal, sinal = trig_angle(self.res.alpha)
            cosbt, sinbt = trig_angle(self.res.beta)
            vfs = Vector(-V*cosal*sinbt, -V*cosbt, -V*sinal*sinbt)
            ofs = Vector(-2*V*cosal*(cosbt*qco2V/c + pbo2V*sinbt/b),
                         2*V*(cosbt*pbo2V/b - qco2V*sinbt/c),
                         -2*V*sinal*(cosbt*qco2V/c + pbo2V*sinbt/b))
            sigbeta = self.res.unsig[:, 0]*vfs+self.res.unsig[:, 1]*ofs
            mubeta = self.res.unmu[:, 0]*vfs+self.res.unmu[:, 1]*ofs
            self._beta = NearFieldResult(self.res, sig=sigbeta, mu=mubeta)
        return self._beta
    @property
    def pbo2V(self):
        if self._pbo2V is None:
            pqr = Vector(2*self.res.speed/self.res.sys.bref, 0.0, 0.0)
            ofs = self.res.wcs.vector_to_global(pqr)
            sigpbo2V = self.res.unsig[:, 1]*ofs
            mupbo2V = self.res.unmu[:, 1]*ofs
            self._pbo2V = NearFieldResult(self.res, sig=sigpbo2V, mu=mupbo2V)
        return self._pbo2V
    @property
    def qco2V(self):
        if self._qco2V is None:
            pqr = Vector(0.0, 2*self.res.speed/self.res.sys.cref, 0.0)
            ofs = self.res.wcs.vector_to_global(pqr)
            sigqco2V = self.res.unsig[:, 1]*ofs
            muqco2V = self.res.unmu[:, 1]*ofs
            self._qco2V = NearFieldResult(self.res, sig=sigqco2V, mu=muqco2V)
        return self._qco2V
    @property
    def rbo2V(self):
        if self._rbo2V is None:
            pqr = Vector(0.0, 0.0, 2*self.res.speed/self.res.sys.bref)
            ofs = self.res.wcs.vector_to_global(pqr)
            sigrbo2V = self.res.unsig[:, 1]*ofs
            murbo2V = self.res.unmu[:, 1]*ofs
            self._rbo2V = NearFieldResult(self.res, sig=sigrbo2V, mu=murbo2V)
        return self._rbo2V
    @property
    def pdbo2V(self):
        if self._pdbo2V is None:
            ofs = Vector(2*self.res.speed/self.res.sys.bref, 0.0, 0.0)
            sigpdbo2V = self.res.unsig[:, 1]*ofs
            mupdbo2V = self.res.unmu[:, 1]*ofs
            self._pbo2V = NearFieldResult(self.res, sig=sigpdbo2V, mu=mupdbo2V)
        return self._pbo2V
    @property
    def qdco2V(self):
        if self._qdco2V is None:
            ofs = Vector(0.0, 2*self.res.speed/self.res.sys.cref, 0.0)
            sigqdco2V = self.res.unsig[:, 1]*ofs
            muqdco2V = self.res.unmu[:, 1]*ofs
            self._qdco2V = NearFieldResult(self.res, sig=sigqdco2V, mu=muqdco2V)
        return self._qdco2V
    @property
    def rdbo2V(self):
        if self._rdbo2V is None:
            ofs = Vector(0.0, 0.0, 2*self.res.speed/self.res.sys.bref)
            sigrdbo2V = self.res.unsig[:, 1]*ofs
            murdbo2V = self.res.unmu[:, 1]*ofs
            self._rdbo2V = NearFieldResult(self.res, sig=sigrdbo2V, mu=murdbo2V)
        return self._rdbo2V
    def neutral_point(self):
        dCzdal = self.alpha.Cz
        dCmdal = self.alpha.Cm
        dxoc = dCmdal/dCzdal
        return self.res.rcg.x-dxoc*self.res.sys.cref
    def system_aerodynamic_matrix(self):
        A = zeros((6, 6))
        F = self.u.nffrctot
        A[0, 0], A[1, 0], A[2, 0] = F.x, F.y, F.z
        M = self.res.wcs.vector_to_local(self.u.nfmomtot)
        A[3, 0], A[4, 0], A[5, 0] = M.x, M.y, M.z
        F = self.v.nffrctot
        A[0, 1], A[1, 1], A[2, 1] = F.x, F.y, F.z
        M = self.res.wcs.vector_to_local(self.v.nfmomtot)
        A[3, 1], A[4, 1], A[5, 1] = M.x, M.y, M.z
        F = self.w.nffrctot
        A[0, 2], A[1, 2], A[2, 2] = F.x, F.y, F.z
        M = self.res.wcs.vector_to_local(self.w.nfmomtot)
        A[3, 2], A[4, 2], A[5, 2] = M.x, M.y, M.z
        F = self.p.nffrctot
        A[0, 3], A[1, 3], A[2, 3] = F.x, F.y, F.z
        M = self.res.wcs.vector_to_local(self.p.nfmomtot)
        A[3, 3], A[4, 3], A[5, 3] = M.x, M.y, M.z
        F = self.q.nffrctot
        A[0, 4], A[1, 4], A[2, 4] = F.x, F.y, F.z
        M = self.res.wcs.vector_to_local(self.q.nfmomtot)
        A[3, 4], A[4, 4], A[5, 4] = M.x, M.y, M.z
        F = self.r.nffrctot
        A[0, 5], A[1, 5], A[2, 5] = F.x, F.y, F.z
        M = self.res.wcs.vector_to_local(self.r.nfmomtot)
        A[3, 5], A[4, 5], A[5, 5] = M.x, M.y, M.z
        return A
    @property
    def stability_derivatives(self):
        from py2md.classes import MDTable, MDHeading, MDReport
        from . import sfrm
        report = MDReport()
        heading = MDHeading('Stability Derivatives', 2)
        report.add_object(heading)
        table = MDTable()
        table.add_column('CLa', sfrm, data=[self.alpha.CL])
        table.add_column('CYa', sfrm, data=[self.alpha.CY])
        table.add_column('Cla', sfrm, data=[self.alpha.Cl])
        table.add_column('Cma', sfrm, data=[self.alpha.Cm])
        table.add_column('Cna', sfrm, data=[self.alpha.Cn])
        report.add_object(table)
        table = MDTable()
        table.add_column('CLb', sfrm, data=[self.beta.CL])
        table.add_column('CYb', sfrm, data=[self.beta.CY])
        table.add_column('Clb', sfrm, data=[self.beta.Cl])
        table.add_column('Cmb', sfrm, data=[self.beta.Cm])
        table.add_column('Cnb', sfrm, data=[self.beta.Cn])
        report.add_object(table)
        table = MDTable()
        table.add_column('CLp', sfrm, data=[self.pbo2V.CL])
        table.add_column('CYp', sfrm, data=[self.pbo2V.CY])
        table.add_column('Clp', sfrm, data=[self.pbo2V.Cl])
        table.add_column('Cmp', sfrm, data=[self.pbo2V.Cm])
        table.add_column('Cnp', sfrm, data=[self.pbo2V.Cn])
        report.add_object(table)
        table = MDTable()
        table.add_column('CLq', sfrm, data=[self.qco2V.CL])
        table.add_column('CYq', sfrm, data=[self.qco2V.CY])
        table.add_column('Clq', sfrm, data=[self.qco2V.Cl])
        table.add_column('Cmq', sfrm, data=[self.qco2V.Cm])
        table.add_column('Cnq', sfrm, data=[self.qco2V.Cn])
        report.add_object(table)
        table = MDTable()
        table.add_column('CLr', sfrm, data=[self.rbo2V.CL])
        table.add_column('CYr', sfrm, data=[self.rbo2V.CY])
        table.add_column('Clr', sfrm, data=[self.rbo2V.Cl])
        table.add_column('Cmr', sfrm, data=[self.rbo2V.Cm])
        table.add_column('Cnr', sfrm, data=[self.rbo2V.Cn])
        report.add_object(table)
        return report
    @property
    def stability_derivatives_body(self):
        from py2md.classes import MDTable, MDHeading, MDReport
        from . import sfrm
        report = MDReport()
        heading = MDHeading('Stability Derivatives Body Axis', 2)
        report.add_object(heading)
        table = MDTable()
        table.add_column('Cxu', sfrm, data=[self.u.Cx])
        table.add_column('Cyu', sfrm, data=[self.u.Cy])
        table.add_column('Czu', sfrm, data=[self.u.Cz])
        table.add_column('Clu', sfrm, data=[self.u.Cmx])
        table.add_column('Cmu', sfrm, data=[self.u.Cmy])
        table.add_column('Cnu', sfrm, data=[self.u.Cmz])
        report.add_object(table)
        table = MDTable()
        table.add_column('Cxv', sfrm, data=[self.v.Cx])
        table.add_column('Cyv', sfrm, data=[self.v.Cy])
        table.add_column('Czv', sfrm, data=[self.v.Cz])
        table.add_column('Clv', sfrm, data=[self.v.Cmx])
        table.add_column('Cmv', sfrm, data=[self.v.Cmy])
        table.add_column('Cnv', sfrm, data=[self.v.Cmz])
        report.add_object(table)
        table = MDTable()
        table.add_column('Cxw', sfrm, data=[self.w.Cx])
        table.add_column('Cyw', sfrm, data=[self.w.Cy])
        table.add_column('Czw', sfrm, data=[self.w.Cz])
        table.add_column('Clw', sfrm, data=[self.w.Cmx])
        table.add_column('Cmw', sfrm, data=[self.w.Cmy])
        table.add_column('Cnw', sfrm, data=[self.w.Cmz])
        report.add_object(table)
        table = MDTable()
        table.add_column('Cxp', sfrm, data=[self.pdbo2V.Cx])
        table.add_column('Cyp', sfrm, data=[self.pdbo2V.Cy])
        table.add_column('Czp', sfrm, data=[self.pdbo2V.Cz])
        table.add_column('Clp', sfrm, data=[self.pdbo2V.Cmx])
        table.add_column('Cmp', sfrm, data=[self.pdbo2V.Cmy])
        table.add_column('Cnp', sfrm, data=[self.pdbo2V.Cmz])
        report.add_object(table)
        table = MDTable()
        table.add_column('Cxq', sfrm, data=[self.qdco2V.Cx])
        table.add_column('Cyq', sfrm, data=[self.qdco2V.Cy])
        table.add_column('Czq', sfrm, data=[self.qdco2V.Cz])
        table.add_column('Clq', sfrm, data=[self.qdco2V.Cmx])
        table.add_column('Cmq', sfrm, data=[self.qdco2V.Cmy])
        table.add_column('Cnq', sfrm, data=[self.qdco2V.Cmz])
        report.add_object(table)
        table = MDTable()
        table.add_column('Cxr', sfrm, data=[self.rdbo2V.Cx])
        table.add_column('Cyr', sfrm, data=[self.rdbo2V.Cy])
        table.add_column('Czr', sfrm, data=[self.rdbo2V.Cz])
        table.add_column('Clr', sfrm, data=[self.rdbo2V.Cmx])
        table.add_column('Cmr', sfrm, data=[self.rdbo2V.Cmy])
        table.add_column('Cnr', sfrm, data=[self.rdbo2V.Cmz])
        report.add_object(table)
        return report
    def __str__(self):
        return self.stability_derivatives._repr_markdown_()
    def _repr_markdown_(self):
        return self.__str__()

def fix_zero(value: float, tol: float=1e-8):
    if abs(value) < tol:
        value = 0.0
    return value
