#%%
# Import Dependencies
from numpy import zeros
from pyapm.tools.naca4 import NACA4
from pygeom.geom3d import Vector
from pygeom.tools.spacing import full_cosine_spacing

from pyapm.classes import Grid, Panel, PanelResult, PanelSystem
from pyapm.outputs.msh import panelresult_to_msh

#%%
# Create Grids
xznum = 20
naca4 = NACA4('0012', cnum = xznum)
x, z = naca4.x, naca4.y

ynum = 30
cdst = full_cosine_spacing(ynum - 1)

ymin = -3.0
ymax = 3.0
yrng = ymax-ymin

y = [ymin+yrng*cdsti for cdsti in cdst]

grids = {}
gidmat = zeros((ynum, xznum*2+1), dtype=int)

gid = 0
for j, (xj, zj) in enumerate(zip(x, z)):
    te = False
    if j == 0 or j == 2*xznum:
        te = True
    for i, yi in enumerate(y):
        gid += 1
        grids[gid] = Grid(gid, xj, yi, zj, te)
        gidmat[i, j] = gid

#%%
# Create Panels
panels = {}
pid = 0
for i in range(2*xznum):
    for j in range(ynum-1):
        pid += 1
        gids = [gidmat[j+1, i], gidmat[j, i], gidmat[j, i+1], gidmat[j+1, i+1]]
        pnlgrds = [grids[gidi] for gidi in gids]
        panels[pid] = Panel(pid, pnlgrds)

# # Close Trailing Edge
# for j in range(ynum-1):
#     pid += 1
#     gids = [gidmat[j+1, -1], gidmat[j, -1], gidmat[j, 0], gidmat[j+1, 0]]
#     pnlgrds = [grids[gidi] for gidi in gids]
#     panels[pid] = Panel(pid, pnlgrds)

n = 2*xznum
# Close Ends
for i in range(xznum):
    pid += 1
    if i == xznum-1:
        gids = [gidmat[0, i+1], gidmat[0, i], gidmat[0, n-i]]
    else:
        gids = [gidmat[0, i+1], gidmat[0, i], gidmat[0, n-i], gidmat[0, n-i-1]]
    pnlgrds = [grids[gidi] for gidi in gids]
    panels[pid] = Panel(pid, pnlgrds)

for i in range(xznum):
    pid += 1
    if i == xznum-1:
        gids = [gidmat[-1, i+1], gidmat[-1, i], gidmat[-1, n-i]]
    else:
        gids = [gidmat[-1, i+1], gidmat[-1, i], gidmat[-1, n-i], gidmat[-1, n-i-1]]
    gids.reverse()
    pnlgrds = [grids[gidi] for gidi in gids]
    panels[pid] = Panel(pid, pnlgrds)

#%%
# Create Panel System
name = 'Test Simple Wing'
bref = yrng
cref = 1.0
sref = bref*cref
rref = Vector(0.25, 0.0, 0.0)
psys = PanelSystem(name, bref, cref, sref, rref)
psys.set_mesh(grids, panels)

psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%%
# Solve Panel Result
alpha = 20.0

pres = PanelResult(f'AoA = {alpha:.1f} degrees', psys)
pres.set_state(alpha = alpha)

#%%
# Output MSH File
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
