from typing import TYPE_CHECKING, Optional

from matplotlib.pyplot import figure
from py2md.classes import MDHeading, MDReport, MDTable
from pygeom.geom3d import Vector
from pygeom.geom3d import zero_vector

from . import PanelResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pygeom.geom3d import Vector

    from .surfacestructure import SurfaceStructure

class SurfaceLoad():
    pres: PanelResult = None
    strc: 'SurfaceStructure' = None
    sf: float = None
    ptfrc: 'Vector' = None
    ptmom: 'Vector' = None
    frctot: Vector = None
    momtot: Vector = None
    frcmin: Vector = None
    mommin: Vector = None
    frcmax: Vector = None
    mommax: Vector = None

    def __init__(self, pres: PanelResult, strc: 'SurfaceStructure',
                 sf: float=1.0) -> None:
        self.pres = pres
        self.strc = strc
        self.sf = sf
        self.update()

    @property
    def rref(self) -> Vector:
        return self.strc.rref

    def update(self) -> None:
        strpres = self.pres.strpres
        self.frctot = Vector(0.0, 0.0, 0.0)
        self.momtot = Vector(0.0, 0.0, 0.0)
        for i, strp in enumerate(self.strc.strps):
            ind = strp.ind
            rrel = strp.point - self.rref
            self.frctot += strpres.stfrc[ind, 0]
            self.momtot += strpres.stmom[ind, 0] + rrel.cross(strpres.stfrc[ind, 0])
        self.rfrc, self.rmom = self.strc.rbdy.return_reactions(self.frctot, self.momtot)
        self.ptfrc = zero_vector(self.strc.pnts.shape, dtype=float)
        self.ptmom = zero_vector(self.strc.pnts.shape, dtype=float)
        ptfrcb = Vector(0.0, 0.0, 0.0)
        ptmomb = Vector(0.0, 0.0, 0.0)
        for i, strp in enumerate(self.strc.strps):
            ind = strp.ind
            inda = 2*i
            indb = inda + 1
            ptfrca = ptfrcb
            ptmoma = ptmomb
            if inda in self.strc.pntinds:
                ptfrca -= self.rfrc[self.strc.pntinds[inda]]
                ptmoma -= self.rmom[self.strc.pntinds[inda]]
            ptfrcb = ptfrca - strpres.stfrc[ind, 0]
            ptmomb = ptmoma - strpres.stmom[ind, 0]
            rrel = strp.point - self.strc.pnts[indb, 0]
            ptmomb -= rrel.cross(strpres.stfrc[ind, 0])
            rrel = self.strc.pnts[indb, 0] - self.strc.pnts[inda, 0]
            ptmomb -= rrel.cross(ptfrca)
            self.ptfrc[inda, 0] = ptfrca
            self.ptfrc[indb, 0] = ptfrcb
            self.ptmom[inda, 0] = ptmoma
            self.ptmom[indb, 0] = ptmomb
        fx = self.ptfrc.x.transpose().tolist()[0]
        fy = self.ptfrc.y.transpose().tolist()[0]
        fz = self.ptfrc.z.transpose().tolist()[0]
        mx = self.ptmom.x.transpose().tolist()[0]
        my = self.ptmom.y.transpose().tolist()[0]
        mz = self.ptmom.z.transpose().tolist()[0]
        self.frcmin = Vector(min(fx), min(fy), min(fz))
        self.frcmax = Vector(max(fx), max(fy), max(fz))
        self.mommin = Vector(min(mx), min(my), min(mz))
        self.mommax = Vector(max(mx), max(my), max(mz))

    def plot_forces(self, ax: Optional['Axes']=None) -> 'Axes':
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
        fx = self.ptfrc.x.transpose().tolist()[0]
        fy = self.ptfrc.y.transpose().tolist()[0]
        fz = self.ptfrc.z.transpose().tolist()[0]
        if self.strc.axis == 'y':
            yp = self.strc.ypos
            ax.plot(yp, fx, label=f'{self.pres.name:s} Vx')
            ax.plot(yp, fy, label=f'{self.pres.name:s} Fy')
            ax.plot(yp, fz, label=f'{self.pres.name:s} Vz')
        elif self.strc.axis == 'z':
            zp = self.strc.zpos
            ax.plot(fx, zp, label=f'{self.pres.name:s} Vx')
            ax.plot(fy, zp, label=f'{self.pres.name:s} Vy')
            ax.plot(fz, zp, label=f'{self.pres.name:s} Fz')
        ax.legend()
        return ax

    def plot_moments(self, ax: Optional['Axes']=None) -> 'Axes':
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
        mx = self.ptmom.x.transpose().tolist()[0]
        my = self.ptmom.y.transpose().tolist()[0]
        mz = self.ptmom.z.transpose().tolist()[0]
        if self.strc.axis == 'y':
            yp = self.strc.ypos
            ax.plot(yp, mx, label=f'{self.pres.name:s} Mx')
            ax.plot(yp, my, label=f'{self.pres.name:s} Ty')
            ax.plot(yp, mz, label=f'{self.pres.name:s} Mz')
        elif self.strc.axis == 'z':
            zp = self.strc.zpos
            ax.plot(mx, zp, label=f'{self.pres.name:s} Mx')
            ax.plot(my, zp, label=f'{self.pres.name:s} My')
            ax.plot(mz, zp, label=f'{self.pres.name:s} Tz')
        ax.legend()
        return ax

    @property
    def point_loads_table(self) -> MDTable:
        table = MDTable()
        table.add_column('#', 'd')
        table.add_column('x', '.3f')
        table.add_column('y', '.3f')
        table.add_column('z', '.3f')
        if self.strc.axis == 'y':
            table.add_column('Vx', '.1f')
            table.add_column('Fy', '.1f')
            table.add_column('Vz', '.1f')
            table.add_column('Mx', '.0f')
            table.add_column('Ty', '.0f')
            table.add_column('Mz', '.0f')
        elif self.strc.axis == 'z':
            table.add_column('Vx', '.1f')
            table.add_column('Vy', '.1f')
            table.add_column('Fz', '.1f')
            table.add_column('Mx', '.0f')
            table.add_column('My', '.0f')
            table.add_column('Tz', '.0f')
        for i in range(self.strc.pnts.shape[0]):
            frc = self.ptfrc[i, 0]
            mom = self.ptmom[i, 0]
            pnt = self.strc.pnts[i, 0]
            x, y, z = pnt.x, pnt.y, pnt.z
            Vx, Fy, Vz = frc.x, frc.y, frc.z
            Mx, Ty, Mz = mom.x, mom.y, mom.z
            table.add_row([i, x, y, z, Vx, Fy, Vz, Mx, Ty, Mz])
        return table

    def to_mdobj(self) -> MDReport:
        report = MDReport()
        heading = MDHeading(f'{self.pres.name} Design Loads for {self.strc.srfc.name}', 1)
        report.add_object(heading)
        table = MDTable()
        table.add_column('Reference', 's')
        table.add_column('x', '.3f')
        table.add_column('y', '.3f')
        table.add_column('z', '.3f')
        table.add_column('Fx', '.1f')
        table.add_column('Fy', '.1f')
        table.add_column('Fz', '.1f')
        table.add_column('Mx', '.1f')
        table.add_column('My', '.1f')
        table.add_column('Mz', '.1f')
        x = self.rref.x
        y = self.rref.y
        z = self.rref.z
        Fx = self.frctot.x
        Fy = self.frctot.y
        Fz = self.frctot.z
        Mx = self.momtot.x
        My = self.momtot.y
        Mz = self.momtot.z
        table.add_row(['Total', x, y, z, Fx, Fy, Fz, Mx, My, Mz])
        report.add_object(table)
        table = MDTable()
        table.add_column('Reaction', 'd')
        table.add_column('x', '.3f')
        table.add_column('y', '.3f')
        table.add_column('z', '.3f')
        table.add_column('Fx', '.1f')
        table.add_column('Fy', '.1f')
        table.add_column('Fz', '.1f')
        table.add_column('Mx', '.1f')
        table.add_column('My', '.1f')
        table.add_column('Mz', '.1f')
        for i, rpnt in enumerate(self.strc.rpnts):
            rfrc = self.rfrc[i]
            rmom = self.rmom[i]
            table.add_row([i+1, rpnt.x, rpnt.y, rpnt.z, rfrc.x, rfrc.y, rfrc.z, rmom.x, rmom.y, rmom.z])
        report.add_object(table)
        table = MDTable()
        table.add_column('Value', 's')
        if self.strc.axis == 'y':
            table.add_column('Vx', '.1f')
            table.add_column('Fy', '.1f')
            table.add_column('Vz', '.1f')
            table.add_column('Mx', '.1f')
            table.add_column('Ty', '.1f')
            table.add_column('Mz', '.1f')
        elif self.strc.axis == 'z':
            table.add_column('Vx', '.1f')
            table.add_column('Vy', '.1f')
            table.add_column('Fz', '.1f')
            table.add_column('Mx', '.1f')
            table.add_column('My', '.1f')
            table.add_column('Tz', '.1f')
        Vx = self.frcmin.x
        Fy = self.frcmin.y
        Vz = self.frcmin.z
        Mx = self.mommin.x
        Ty = self.mommin.y
        Mz = self.mommin.z
        table.add_row(['Min', Vx, Fy, Vz, Mx, Ty, Mz])
        Vx = self.frcmax.x
        Fy = self.frcmax.y
        Vz = self.frcmax.z
        Mx = self.mommax.x
        Ty = self.mommax.y
        Mz = self.mommax.z
        table.add_row(['Max', Vx, Fy, Vz, Mx, Ty, Mz])
        report.add_object(table)
        return report

    def __str__(self) -> str:
        return self.to_mdobj().__str__()

    def _repr_markdown_(self) -> str:
        return self.to_mdobj()._repr_markdown_()
