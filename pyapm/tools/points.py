from typing import TYPE_CHECKING

from numpy import arange, argmin, asarray, logical_not, zeros
from pygeom.geom3d import Vector

from ..classes.panelsystem import PanelSystem

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fetch_pids_ttol(pnts: Vector, psys: PanelSystem, ztol: float=0.01, ttol: float=0.1):
    shp = pnts.shape
    pnts = pnts.reshape((-1, 1))
    numpnt = pnts.shape[0]
    numpnl = len(psys.pnls)
    pidm = zeros((1, numpnl), dtype=int)
    wintm = zeros((numpnt, numpnl), dtype=bool)
    abszm = zeros((numpnt, numpnl))
    if isinstance(psys.pnls, dict):
        for pnl in psys.pnls.values():
            pidm[0, pnl.ind] = pnl.pid
            wintm[:, pnl.ind], abszm[:, pnl.ind] = pnl.within_and_absz_ttol(pnts[:, 0], ttol=ttol)
    elif isinstance(psys.pnls, list):
        for pnl in psys.pnls:
            pidm[0, pnl.ind] = pnl.pid
            wintm[:, pnl.ind], abszm[:, pnl.ind] = pnl.within_and_absz_ttol(pnts[:, 0], ttol=ttol)
    abszm[wintm is False] = float('inf')
    minm = argmin(abszm, axis=1)
    minm = asarray(minm).flatten()
    pidm = asarray(pidm).flatten()
    pids = pidm[minm]
    pids = asarray([pids], dtype=int).transpose()
    indp = arange(numpnt)
    minz = asarray(abszm[indp, minm]).flatten()
    minz = asarray([minz]).transpose()
    chkz = minz < ztol
    pids[logical_not(chkz)] = 0
    pids = pids.reshape(shp)
    chkz = chkz.reshape(shp)
    return pids, chkz

def point_results(pnts: Vector, psys: PanelSystem, pids: 'NDArray', chkz: 'NDArray',
                  pnlres: 'NDArray', ttol: float=0.1):
    res = zeros(pnts.shape)
    for i in range(pnts.shape[0]):
        for j in range(pnts.shape[1]):
            if chkz[i, j]:
                pid = pids[i, j]
                pnl = psys.pnls[pid]
                pnt = pnts[i, j]
                res[i, j] = pnl.point_res(pnlres, pnt, ttol=ttol)
    return res
