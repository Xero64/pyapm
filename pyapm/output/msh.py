from ..classes.panelresult import PanelResult
from numpy.matlib import matrix

def result_to_msh(pres: PanelResult, mshfilepath: str):
    psys = pres.sys
    gidlst = list(psys.grds)
    pidlst = list(psys.pnls)
    tidlst = []
    qidlst = []
    for pid, pnl in psys.pnls.items():
        if pnl.num == 3:
            tidlst.append(pid)
        if pnl.num == 4:
            qidlst.append(pid)
    gidlst.sort()
    tidlst.sort()
    qidlst.sort()
    pidlst.sort()
    mingid = gidlst[0]
    maxgid = gidlst[-1]
    lengid = len(gidlst)
    lentid = len(tidlst)
    lenqid = len(qidlst)
    minpid = pidlst[0]
    maxpid = pidlst[-1]
    lenpid = len(pidlst)
    nelbl = 0
    if lentid > 0:
        nelbl += 1
    if lenqid > 0:
        nelbl += 1
    with open(mshfilepath, 'wt') as mshfile:
        mshfile.write('$MeshFormat\n')
        mshfile.write('4.1 0 8\n')
        mshfile.write('$EndMeshFormat\n')
        mshfile.write('$Nodes\n')
        mshfile.write('{:d} {:d} {:d} {:d}\n'.format(1, lengid, mingid, maxgid))
        mshfile.write('{:d} {:d} {:d} {:d}\n'.format(2, 1, 0, lengid))
        frmstr = '{:d}\n'
        for gid in gidlst:
            mshfile.write(frmstr.format(gid))
        frmstr = '{:} {:} {:}\n'
        for gid in gidlst:
            grd = psys.grds[gid]
            x = grd.x
            y = grd.y
            z = grd.z
            mshfile.write(frmstr.format(x, y, z))
        mshfile.write('$EndNodes\n')
        mshfile.write('$Elements\n')
        mshfile.write('{:d} {:d} {:d} {:d}\n'.format(nelbl, lenpid, minpid, maxpid))
        bl = 1
        if lentid > 0:
            mshfile.write('{:d} {:d} {:d} {:d}\n'.format(2, bl, 2, lentid))
            for pid in tidlst:
                outstr = '{:d}'.format(pid)
                pnl = psys.pnls[pid]
                for grd in pnl.grds:
                    outstr += ' {:d}'.format(grd.gid)
                outstr += '\n'
                mshfile.write(outstr)
        if lenqid > 0:
            mshfile.write('{:d} {:d} {:d} {:d}\n'.format(2, bl, 3, lenqid))
            for pid in qidlst:
                outstr = '{:d}'.format(pid)
                pnl = psys.pnls[pid]
                for grd in pnl.grds:
                    outstr += ' {:d}'.format(grd.gid)
                outstr += '\n'
                mshfile.write(outstr)
        mshfile.write('$EndElements\n')
        optstr = ''
        optstr += 'Mesh.Lines = 0;\n'
        optstr += 'Mesh.LineNumbers = 0;\n'
        optstr += 'Mesh.SurfaceEdges = 0;\n'
        optstr += 'Mesh.SurfaceFaces = 0;\n'
        optstr += 'Mesh.SurfaceNumbers = 0;\n'
        optstr += 'Mesh.VolumeEdges = 0;\n'
        optstr += 'Mesh.VolumeFaces = 0;\n'
        optstr += 'Mesh.VolumeNumbers = 0;\n'
        view = 0
        # Source Strength
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Source Strength"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.sig[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Doublet Strength
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Doublet Strength"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.mu[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Coefficient of Pressure
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Coefficient of Pressure"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.cp[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Velocity Potential
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Velocity Potential"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.phi[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Normal Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Normal Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.vl.z[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Longitudinal Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Longitudinal Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.vl.x[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Transverse Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Transverse Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.vl.y[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Tangential Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Tangential Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.vt[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Normal Force
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Normal Force"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            mshfile.write(frmstr.format(pnl.pid, pres.nrmfrc[pnl.ind, 0]))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
    optfilepath = mshfilepath + '.opt'
    with open(optfilepath, 'wt') as optfile:
        optfile.write(optstr)
