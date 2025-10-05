from typing import TYPE_CHECKING

from numpy import argmin, logical_not, ndenumerate, take_along_axis, zeros
from pygeom.geom3d import Vector

from ..classes.panelsystem import PanelSystem

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fetch_pids_ttol(pnts: Vector, psys: PanelSystem, ztol: float=0.01, ttol: float=0.1):
    numpnl = len(psys.pnls)
    pidm = zeros(numpnl, dtype=int)
    wintm = zeros((numpnl, *pnts.shape), dtype=bool)
    abszm = zeros((numpnl, *pnts.shape))
    for pnl in psys.pnls.values():
        pidm[pnl.ind] = pnl.pid
        wintm[pnl.ind, ...], abszm[pnl.ind, ...] = pnl.within_and_absz_ttol(pnts, ttol=ttol)
    abszm[logical_not(wintm)] = float('inf')
    minm: 'NDArray' = argmin(abszm, axis=0, keepdims=True)
    pids = pidm[minm].reshape(pnts.shape)
    minz = take_along_axis(abszm, minm, axis=0).reshape(pnts.shape)
    chkz = minz < ztol
    return pids, chkz

def point_results(pnts: Vector, psys: PanelSystem, pids: 'NDArray', chkz: 'NDArray',
                  pnlres: 'NDArray', ttol: float=0.1):
    res = zeros(pnts.shape)
    for (ind, chk) in ndenumerate(chkz):
        if chk:
            pnt = pnts[ind]
            pid = pids[ind]
            pnl = psys.pnls[pid]
            res[ind] = pnl.point_res(pnlres, pnt, ttol=ttol)
    return res
