from typing import TYPE_CHECKING

from pygeom.geom3d import Vector

from ..constantcontrol import ConstantControl, ControlObject
from ..constantgeometry import ConstantGeometry
from ..constantpanel import ConstantPanel
from ..constantresult import ConstantResult
from ..constantsystem import ConstantSystem
from ..constantwakepanel import ConstantWakePanel
from ....tools.mass import MassObject

if TYPE_CHECKING:
    from pyvlm.classes import LatticeSystem
    from pyvlm.classes.latticecontrol import LatticeControl
    from pyvlm.classes.latticepanel import LatticePanel


def constant_system_from_lattice_system(lsys: 'LatticeSystem') -> ConstantSystem:

    dirl = Vector(1.0, 0.0, 0.0)

    lpnls: list[list[list[LatticePanel]]] = []
    for i, lsrfc in enumerate(lsys.srfcs):
        lpnls.append([])
        for j, lstrp in enumerate(lsrfc.strps):
            lpnls[i].append([])
            for lpnl in lstrp.pnls:
                lpnls[i][j].append(lpnl)

    lctrl_objs: list['LatticeControl'] = []
    lctrl_pnts: list[Vector] = []
    for srfc in lsys.srfcs:
        for sht in srfc.shts:
            for ctrl in sht.ctrls.values():
                lctrl_objs.append(ctrl)
                ctrl.pnls = set(ctrl.pnls)
                lctrl_pnts.append(sht.sct1.pnt + Vector(ctrl.xhinge*sht.sct1.chord, 0.0, 0.0))

    cctrl_objs: list[ControlObject] = []
    cctrl_obj_lpnl_index: dict[ControlObject, list[tuple[str, int, int]]] = {}
    for i, lctrl in enumerate(lctrl_objs):
        cctrl_obj = ControlObject(lctrl.name)
        cctrl_obj.posgain = lctrl.posgain
        cctrl_obj.neggain = lctrl.neggain
        cctrl_obj.position = lctrl.xhinge
        cctrl_obj.vector = lctrl.uhvec
        cctrl_obj.point = lctrl_pnts[i]
        cctrl_objs.append(cctrl_obj)
        cctrl_obj_lpnl_index[cctrl_obj] = []

    geometry = ConstantGeometry()
    for srfc in lsys.srfcs:
        geomsurf = geometry.add_surface(srfc.name)
        numb = len(srfc.strps)
        numc = len(srfc.cspc)
        geomsurf.points = Vector.zeros((numb + 1, numc + 1))
        geomsurf.ppoints = Vector.zeros((numb, numc))
        geomsurf.pnormals = Vector.zeros((numb, numc))
        for i in range(numb):
            lstrp = srfc.strps[i]
            for j in range(numc):
                lpnl = lstrp.pnls[j]
                geomsurf.points[i, j] = lpnl.pnta
                geomsurf.ppoints[i, j] = lpnl.pntc
                geomsurf.pnormals[i, j] = lpnl.nrml
                for cctrl_obj, lctrl_obj in zip(cctrl_objs, lctrl_objs):
                    if lpnl in lctrl_obj.pnls:
                        cctrl_obj_lpnl_index[cctrl_obj].append((srfc.name, i, j))
            lpnl = lstrp.pnls[-1]
            geomsurf.points[i, -1] = lpnl.pnts[2]
        lstrp = srfc.strps[-1]
        for j in range(numc):
            lpnl = lstrp.pnls[j]
            geomsurf.points[-1, j] = lpnl.pntb
        lpnl = lstrp.pnls[-1]
        geomsurf.points[-1, -1] = lpnl.pnts[3]

    gid = 0
    for srfc in lsys.srfcs:
        geomsurf = geometry.get_surface(srfc.name)
        gid = geomsurf.mesh_grids(gid)

    dpanels: list[ConstantPanel] = []
    npanels: list[ConstantPanel] = []
    wpanels: list[ConstantWakePanel] = []

    pid = 0
    for srfc in lsys.srfcs:
        geomsurf = geometry.get_surface(srfc.name)
        pid = geomsurf.mesh_panels(pid, dirl)
        npanels.extend(geomsurf.npanels)
        wpanels.extend(geomsurf.wpanels)

    csys = ConstantSystem(lsys.name, dpanels, npanels, wpanels)
    csys.cref = lsys.cref
    csys.bref = lsys.bref
    csys.sref = lsys.sref
    csys.CDo = lsys.cdo
    csys.rref = Vector.from_obj(lsys.rref)
    csys.ctrls = {control: ConstantControl(control, csys) for control in lsys.ctrls}
    csys.mass = MassObject.from_dict(lsys.mass.__dict__)

    csys.geometry = geometry

    for cctrl_obj in cctrl_objs:
        for srfcname, i, j in cctrl_obj_lpnl_index[cctrl_obj]:
            geomsurf = geometry.get_surface(srfcname)
            cctrl_obj.panels.append(geomsurf.panels[i, j])
        csys.ctrls[cctrl_obj.name].control_objects.append(cctrl_obj)

    for lres in lsys.results.values():
        cres = ConstantResult(lres.name, csys)
        cres.rho = lres.rho
        cres.speed = lres.speed
        cres.alpha = lres.alpha
        cres.beta = lres.beta
        cres.mach = lres.mach
        cres.pbo2v = lres.pbo2v
        cres.qco2v = lres.qco2v
        cres.rbo2v = lres.rbo2v
        cres.CDo = csys.CDo
        if lres.mass is not None:
            cres.mass = MassObject.from_dict(lres.mass.__dict__)
        for control in lres.ctrls:
            cres.ctrls[control] = lres.ctrls[control]
        csys.results[cres.name] = cres
        if hasattr(lres, 'targets'):
            cres.trim_result.targets = lres.targets
        else:
            targets = {}
            targets['alpha'] = ('alpha', lres.alpha)
            targets['beta'] = ('beta', lres.beta)
            targets['pbo2v'] = ('pbo2v', lres.pbo2v)
            targets['qco2v'] = ('qco2v', lres.qco2v)
            targets['rbo2v'] = ('rbo2v', lres.rbo2v)
            for control in lres.ctrls:
                targets[control] = (control, lres.ctrls[control])
            cres.trim_result.targets = targets

        csys.load_initial_state(lsys.source)

    return csys
