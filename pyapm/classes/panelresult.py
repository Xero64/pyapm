from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from pygeom.matrix3d import elementwise_dot_product, elementwise_multiply, elementwise_cross_product
from numpy.matlib import matrix, ones
from math import cos, sin, radians
from numpy.matlib import sqrt, square, multiply, absolute

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
    _sig: matrix = None
    _mu: matrix = None
    _nfres = None
    _grdres = None
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
    def sig(self):
        if self._sig is None:
            self._sig = self.sys.sig[:, 0]*self.vfs + self.sys.sig[:, 1]*self.ofs
        return self._sig
    @property
    def mu(self):
        if self._mu is None:
            self._mu = self.sys.mu[:, 0]*self.vfs + self.sys.mu[:, 1]*self.ofs
        return self._mu
    @property
    def nfres(self):
        if self._nfres is None:
            self._nfres = NearFieldResult(self)
        return self._nfres
    @property
    def grdres(self):
        if self._grdres is None:
            self._grdres = GridResult(self)
        return self._grdres
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
        outstr = '# Lattice Result '+self.name+' for '+self.sys.name+'\n'
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
    # _nfvg = None
    # _nfvl = None
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
    def __init__(self, res: PanelResult):
        self.res = res
    # @property
    # def nfvg(self):
    #     if self._nfvg is None:
    #         self._nfvg = self.res.vfs + self.res.sys.avs*self.res.sig + self.res.sys.avm*self.res.mu
    #     return self._nfvg
    # @property
    # def nfvl(self):
    #     if self._nfvl is None:
    #         self._nfvl = zero_matrix_vector(self.nfvg.shape, dtype=float)
    #         for pnl in self.res.sys.pnls.values():
    #             self._nfvl[pnl.ind, 0] = pnl.crd.vector_to_local(self.nfvg[pnl.ind, 0])
    #     return self._nfvl
    @property
    def nfql(self):
        if self._nfql is None:
            self._nfql = zero_matrix_vector((self.res.sys.numpnl, 1), dtype=float)
            for pnl in self.res.sys.pnls.values():
                qx, qy = pnl.diff_mu(self.res.mu)
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
            self._nfphi = self.res.sys.apm*self.res.mu + self.res.sys.aps*self.res.sig
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

class GridResult(object):
    res: PanelResult = None
    _grdphi: MatrixVector = None
    _grdvg: MatrixVector = None
    def __init__(self, res: PanelResult):
        self.res = res
    @property
    def grdphi(self):
        if self._grdphi is None:
            self._grdphi = self.res.sys.gpm*self.res.mu + self.res.sys.gps*self.res.sig
        return self._grdphi
    @property
    def grdvg(self):
        if self._grdvg is None:
            self._grdvg = self.res.sys.gvm*self.res.mu + self.res.sys.gvs*self.res.sig + self.res.vfs
        return self._grdvg

class FarFieldResult(object):
    def __init__(self):
        pass

def fix_zero(value: float, tol: float=1e-8):
    if abs(value) < tol:
        value = 0.0
    return value
