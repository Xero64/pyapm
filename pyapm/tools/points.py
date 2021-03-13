from typing import List
from pygeom.matrix3d import MatrixVector
from numpy.matlib import zeros, argmin, array, arange, logical_not, matrix
from ..classes.panelsystem import PanelSystem
from ..classes.panel import Panel

def fetch_pids(pnts: MatrixVector, psys: PanelSystem, ztol: float=0.1):
    shp = pnts.shape
    pnts = pnts.reshape((-1, 1))
    numpnt = pnts.shape[0]
    numpnl = len(psys.pnls)
    pidm = zeros((1, numpnl), dtype=int)
    thm = zeros((numpnt, numpnl), dtype=float)
    abszm = zeros((numpnt, numpnl), dtype=float)
    for pnl in psys.pnls.values():
        pidm[0, pnl.ind] = pnl.pid
        thm[:, pnl.ind], abszm[:, pnl.ind] = pnl.within_and_absz(pnts[:, 0])
    abszm[thm == 0.0] = float('inf')
    minm = argmin(abszm, axis=1)
    minm = array(minm).flatten()
    pidm = array(pidm).flatten()
    pids = pidm[minm]
    pids = matrix([pids], dtype=int).transpose()
    indp = arange(numpnt)
    minz = array(abszm[indp, minm]).flatten()
    minz = matrix([minz], dtype=float).transpose()
    chkz = minz < ztol
    pids[logical_not(chkz)] = 0
    pids = pids.reshape(shp)
    chkz = chkz.reshape(shp)
    return pids, chkz

def point_results(pnts: MatrixVector, psys: PanelSystem, pids: array, chkz: array, pnlres: matrix, ztol: float=1e-8):
    res = zeros(pnts.shape, dtype=float)
    for i in range(pnts.shape[0]):
        for j in range(pnts.shape[1]):
            if chkz[i, j]:
                pid = pids[i, j]
                pnl = psys.pnls[pid]
                pnt = pnts[i, j]
                res[i, j] = pnl.point_res(pnlres, pnt)
    return res
