#%%
# Import Dependencies
from IPython.display import display_markdown
from numpy import cos, radians, sin
from pyapm.methods.cpm import ConstantGrid
from pyapm.methods.cpm import (ConstantPanel, ConstantPlot, ConstantResult,
                                 ConstantSystem, ConstantWakePanel)
from pygeom.geom3d import BSplineSurface, Vector
from pygeom.tools.k3d import Plot, k3d_surface
from pygeom.tools.spacing import full_cosine_spacing

#%%
# Create Surface
span = 10.0
root_chord = 0.5
tip_chord = 0.5
sweep = 0.0
dihedral = 0.0
perc = 0.25
sweep = radians(sweep)
dihedral = radians(dihedral)

ctlpnts = Vector.zeros((2, 3))
ctlpnts[0, 0] = Vector(-perc*tip_chord + span/2*sin(sweep), -span/2*cos(dihedral), span/2*sin(dihedral))
ctlpnts[1, 0] = Vector((1.0-perc)*tip_chord + span/2*sin(sweep), -span/2*cos(dihedral), span/2*sin(dihedral))
ctlpnts[0, 1] = Vector(-perc*root_chord, 0.0, 0.0)
ctlpnts[1, 1] = Vector((1.0-perc)*root_chord, 0.0, 0.0)
ctlpnts[0, 2] = Vector(-perc*tip_chord + span/2*sin(sweep), span/2*cos(dihedral), span/2*sin(dihedral))
ctlpnts[1, 2] = Vector((1.0-perc)*tip_chord + span/2*sin(sweep), span/2*cos(dihedral), span/2*sin(dihedral))

surface = BSplineSurface(ctlpnts, udegree=1, vdegree=1)

plot = Plot()
plot += k3d_surface(surface)
plot.display()

mac = 0.5*(root_chord + tip_chord)
area = span*mac

#%%
# Create Panels
numc = 13
numb = 51

cspc = full_cosine_spacing(numc - 1)
# bspc = full_cosine_spacing((numb - 1) // 2)
# print(f'bspc.shape = {bspc.shape}\n')
# bspc = concatenate((bspc[:-1], bspc + 1.0))/2
bspc = full_cosine_spacing(numb - 1)
# print(f'cspc = {cspc}\n')
# print(f'bspc = {bspc}\n')

points = surface.evaluate_points_at_uv(cspc, bspc).transpose()

dirw = Vector(1.0, 0.0, 0.0)
# print(f'dirw = {dirw:.6f}\n')

grids: list[list[ConstantGrid]] = []
k = 0
for i in range(points.shape[0]):
    grids.append([])
    for j in range(points.shape[1]):
        grid = ConstantGrid(k, *points[i, j].to_xyz())
        # grid.ind = k
        grids[i].append(grid)
        k += 1

ppoints = Vector.zeros((numb - 1, numc - 1))

panels: list[list[ConstantPanel]] = []

dpanels: list[ConstantPanel] = []

npanels: list[ConstantPanel] = []
wpanels: list[ConstantWakePanel] = []
k = 0
for i in range(numb - 1):
    panels.append([])
    for j in range(numc - 1):
        grida = grids[i + 1][j]
        gridb = grids[i][j]
        gridc = grids[i][j + 1]
        gridd = grids[i + 1][j + 1]
        npanel = ConstantPanel(k, grida, gridb, gridc, gridd)
        k += 1
        npanels.append(npanel)
        panels[i].append(npanel)
        ppoints[i, j] = npanel.point
    grida = grids[i][-1]
    gridb = grids[i + 1][-1]
    wpanel = ConstantWakePanel(k, grida, gridb, dirw)
    k += 1
    wpanels.append(wpanel)

sysname = 'Simple System'
sys = ConstantSystem(sysname, dpanels, npanels, wpanels)
sys.cref = mac
sys.bref = span
sys.sref = area
display_markdown(sys)

resname = 'Simple Result'
res = ConstantResult(resname, sys)
res.set_state(alpha=3.0)
display_markdown(res)

# fig = figure(figsize=(10, 10))
# ax = fig.gca()
# ax.invert_yaxis()
# ax.set_aspect('equal')
# cf = ax.contourf(points.y, points.x, gvg.z)
# fig.colorbar(cf, location='bottom')
# _ = ax.set_title('Grid Velocity in Z')

# fig = figure(figsize=(10, 10))
# ax = fig.gca()
# ax.invert_yaxis()
# ax.set_aspect('equal')
# cf = ax.contourf(points.y, points.x, gfrc.z)
# fig.colorbar(cf, location='bottom')
# _ = ax.set_title('Grid Force in Z')

# pvp = vp[panel_index]
# pmun = mun[panel_index]

# fig = figure(figsize=(10, 10))
# ax = fig.gca()
# ax.invert_yaxis()
# ax.set_aspect('equal')
# cf = ax.contourf(ppoints.y, ppoints.x, pvp.z)
# fig.colorbar(cf, location='bottom')
# _ = ax.set_title('Panel Velocity in Z')

# fig = figure(figsize=(10, 10))
# ax = fig.gca()
# ax.invert_yaxis()
# ax.set_aspect('equal')
# cf = ax.contourf(ppoints.y, ppoints.x, pmun)
# fig.colorbar(cf, location='bottom')
# _ = ax.set_title('Panel Mu')

#%%
# Create Plot
cplot = ConstantPlot(sys, res)
cplot.calculate_dpanel()
cplot.calculate_npanel()

plot = cplot.create_plot()
plot += cplot.npanel_mesh_plot(res.mun)
plot.display()


# fgrids = sys.gridvec
# forces = res.result.ngfrc.ravel()*100.0

# print(f'{fgrids.shape = }')
# print(f'{forces.shape = }')

# frcvecs = vectors(fgrids.stack_xyz().astype('float32'), forces.stack_xyz().astype('float32'), head_size=0.1, line_width=0.001)
# # velvecs = vectors(points.stack_xyz().astype('float32'), vg.stack_xyz().astype('float32'))

# # pnrmvecs = vectors(sys.npoints.stack_xyz().astype('float32'), sys.nnormal.stack_xyz().astype('float32'))

# plot = Plot()
# plot += k3d_surface(surface, unum=13, vnum=13)
# plot += frcvecs
# # plot += velvecs
# # plot += pnrmvecs
# plot.display()

