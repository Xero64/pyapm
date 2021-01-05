from .panel import Panel
from .panelprofile import PanelProfile
from typing import List

class PanelStrip(object):
    prfa: PanelProfile = None
    prfb: PanelProfile = None
    sht: object = None
    pnls: List[Panel] = None
    def __init__(self, prfa: PanelProfile, prfb: PanelProfile, sht: object):
        self.prfa = prfa
        self.prfb = prfb
        self.sht = sht
    @property
    def noload(self):
        return self.sht.noload
    def mesh_panels(self, pid: int):
        num = len(self.prfa.grds)-1
        self.pnls = []
        for i in range(num):
            grd1 = self.prfa.grds[i]
            grd2 = self.prfa.grds[i+1]
            grd3 = self.prfb.grds[i+1]
            grd4 = self.prfb.grds[i]
            gids = [grd1.gid, grd2.gid, grd3.gid, grd4.gid]
            pnl = Panel(pid, gids)
            pnl.noload = self.noload
            self.pnls.append(pnl)
            pid += 1
        return pid
