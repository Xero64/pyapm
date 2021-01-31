# from os.path import dirname, join
from ..classes.panelresult import PanelResult
from ..classes.panelsystem import PanelSystem

def panelresult_to_msh(pres: PanelResult, mshfilepath: str):
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
        # Panel Source Strength
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Source Strength"\n')
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
        # Panel Doublet Strength
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Doublet Strength"\n')
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
        # Panel Velocity Potential
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Velocity Potential"\n')
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
        # Panel Longitudinal Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Longitudinal Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            val = pres.qloc.x[pnl.ind, 0]
            if pnl.sct is None:
                if val < minval:
                    minval = val
                if val > maxval:
                    maxval = val
            mshfile.write(frmstr.format(pnl.pid, val))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Panel Transverse Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Transverse Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            val = pres.qloc.y[pnl.ind, 0]
            if pnl.sct is None:
                if val < minval:
                    minval = val
                if val > maxval:
                    maxval = val
            mshfile.write(frmstr.format(pnl.pid, val))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Panel Surface Velocity
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Surface Velocity"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            val = pres.qs[pnl.ind, 0]
            if pnl.sct is None:
                if val < minval:
                    minval = val
                if val > maxval:
                    maxval = val
            mshfile.write(frmstr.format(pnl.pid, val))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Panel Coefficient of Pressure
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Coefficient of Pressure"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            val = pres.nfres.nfcp[pnl.ind, 0]
            if pnl.sct is None:
                if val < minval:
                    minval = val
                if val > maxval:
                    maxval = val
            mshfile.write(frmstr.format(pnl.pid, val))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Panel Normal Pressure
        mshfile.write('$ElementData\n')
        mshfile.write('1\n')
        mshfile.write('"Panel Normal Pressure"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:}\n'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            val = pres.nfres.nfprs[pnl.ind, 0]
            if pnl.sct is None:
                if val < minval:
                    minval = val
                if val > maxval:
                    maxval = val
            mshfile.write(frmstr.format(pnl.pid, val))
        mshfile.write('$EndElementData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Grid Source Strength
        mshfile.write('$ElementNodeData\n')
        mshfile.write('1\n')
        mshfile.write('"Grid Source Strength"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:d}'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            vals = pnl.grid_res(pres.sig)
            numv = len(vals)
            mshfile.write(frmstr.format(pnl.pid, numv))
            for val in vals:
                mshfile.write(' {:}'.format(val))
            mshfile.write('\n')
        mshfile.write('$EndElementNodeData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Grid Doublet Strength
        mshfile.write('$ElementNodeData\n')
        mshfile.write('1\n')
        mshfile.write('"Grid Doublet Strength"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:d}'
        for pid in pidlst:
            pnl = psys.pnls[pid]
            vals = pnl.grid_res(pres.mu)
            numv = len(vals)
            mshfile.write(frmstr.format(pnl.pid, numv))
            for val in vals:
                mshfile.write(' {:}'.format(val))
            mshfile.write('\n')
        mshfile.write('$EndElementNodeData\n')
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].RangeType = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Grid Coefficient of Pressure
        mshfile.write('$ElementNodeData\n')
        mshfile.write('1\n')
        mshfile.write('"Grid Coefficient of Pressure"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:d}'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            vals = pnl.grid_res(pres.nfres.nfcp)
            numv = len(vals)
            minvals = min(vals)
            maxvals = max(vals)
            if pnl.sct is None:
                if minvals < minval:
                    minval = minvals
                if maxvals > maxval:
                    maxval = maxvals
            mshfile.write(frmstr.format(pnl.pid, numv))
            for val in vals:
                mshfile.write(' {:}'.format(val))
            mshfile.write('\n')
        mshfile.write('$EndElementNodeData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
        # Grid Normal Pressure
        mshfile.write('$ElementNodeData\n')
        mshfile.write('1\n')
        mshfile.write('"Grid Normal Pressure"\n')
        mshfile.write('1\n')
        mshfile.write('0.0\n')
        mshfile.write('3\n')
        mshfile.write('0\n')
        mshfile.write('1\n')
        mshfile.write('{:d}\n'.format(lenpid))
        frmstr = '{:d} {:d}'
        maxval = float('-inf')
        minval = float('+inf')
        for pid in pidlst:
            pnl = psys.pnls[pid]
            vals = pnl.grid_res(pres.nfres.nfprs)
            numv = len(vals)
            minvals = min(vals)
            maxvals = max(vals)
            if pnl.sct is None:
                if minvals < minval:
                    minval = minvals
                if maxvals > maxval:
                    maxval = maxvals
            mshfile.write(frmstr.format(pnl.pid, numv))
            for val in vals:
                mshfile.write(' {:}'.format(val))
            mshfile.write('\n')
        mshfile.write('$EndElementNodeData\n')
        optstr += 'View[{:d}].RangeType = 2;\n'.format(view)
        optstr += 'View[{:d}].CustomMax = {:};\n'.format(view, maxval)
        optstr += 'View[{:d}].CustomMin = {:};\n'.format(view, minval)
        optstr += 'View[{:d}].Light = 0;\n'.format(view)
        optstr += 'View[{:d}].SaturateValues = 1;\n'.format(view)
        optstr += 'View[{:d}].Visible = 0;\n'.format(view)
        view += 1
    optfilepath = mshfilepath + '.opt'
    with open(optfilepath, 'wt') as optfile:
        optfile.write(optstr)

def panelsystem_to_msh(psys: PanelSystem, mshfilepath: str):
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

def panelresults_to_msh(psys: PanelSystem, outputs: dict):

    # path = dirname(psys.source)

    for case in psys.results:
        pres = psys.results[case]
        for output in outputs[case]:
            output = output.lower()
            if output[-4:] == '.msh':
                # panelresult_to_msh(pres, join(path, output))
                panelresult_to_msh(pres, output)
