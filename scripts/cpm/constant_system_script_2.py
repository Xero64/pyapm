#%%
# Import Dependencies
from time import perf_counter

from IPython.display import display_markdown
from matplotlib.pyplot import figure
from numpy import (cos, diag, linspace, pi, radians, set_printoptions, sin,
                   zeros)
from numpy.typing import NDArray
from pyapm.methods.cpm import ConstantGrid
from pyapm.core.flow import Flow
from pyapm.core.freestream import FreeStream
from pyapm.methods.cpm import (ConstantPanel, ConstantSystem,
                                 ConstantWakePanel)
from pyapm.tools.mesh import point_mesh_xy
from pyapm.tools.plot import point_contourf_xy
from pygeom.geom3d import ParamSurface, Vector
from pygeom.tools.k3d import Plot, k3d_surface, k3d_surface_normals, vectors
from pygeom.tools.spacing import full_cosine_spacing

set_printoptions(precision=6, suppress=True)

from py2md.classes import MDParamTable

#%%
# Create Surface
radius = 1.0

def ruv(u: NDArray, v: NDArray) -> Vector:
    x = -radius*cos(pi*v)
    y = radius*cos(2*pi*u)*sin(pi*v)
    z = radius*sin(2*pi*u)*sin(pi*v)
    return Vector(x, y, z)

def drdv(u: NDArray, v: NDArray) -> Vector:
    x = pi*radius*sin(pi*v)
    y = pi*radius*cos(2*pi*u)*cos(pi*v)
    z = pi*radius*sin(2*pi*u)*cos(pi*v)
    return Vector(x, y, z)

def drdu(u: NDArray, v: NDArray) -> Vector:
    x = zeros(v.shape, dtype=u.dtype)
    y = -2*pi*radius*sin(2*pi*u)*sin(pi*v)
    z = 2*pi*radius*cos(2*pi*u)*sin(pi*v)
    drdu = Vector(x, y, z)

    chk_v0 = v == 0.0
    u_v0 = u[chk_v0]
    v_v0 = v[chk_v0]
    nrml_v0 = Vector(-1.0, 0.0, 0.0)
    drdv_v0 = drdv(u_v0, v_v0)
    drdu_v0 = drdv_v0.cross(nrml_v0)

    chk_v1 = v == 1.0
    u_v1 = u[chk_v1]
    v_v1 = v[chk_v1]
    nrml_v1 = Vector(1.0, 0.0, 0.0)
    drdv_v1 = drdv(u_v1, v_v1)
    drdu_v1 = drdv_v1.cross(nrml_v1)

    drdu[chk_v0] = drdu_v0
    drdu[chk_v1] = drdu_v1

    return drdu

surface = ParamSurface(ruv, drdu, drdv)

plot = Plot()
plot += k3d_surface(surface, unum=13, vnum=13)
plot += k3d_surface_normals(surface, unum=13, vnum=13)
plot.display()

#%%
# Create Panels
cnum = 37
bnum = 37

points = surface.evaluate_points(cnum, bnum).transpose()
normals = surface.evaluate_normals(cnum, bnum).transpose()

print(f'points.shape = \n{points.shape}\n')
cnum = points.shape[0]
bnum = points.shape[1]

dirw = Vector(1.0, 0.0, 0.0)
print(f'dirw = {dirw:.6f}\n')

alpha = 0.0
alrad = radians(alpha)

vfs = Vector(cos(alrad), 0.0, sin(alrad))
vmag = vfs.return_magnitude()
rho = 1.0
q = rho*vmag**2/2

dirx = vfs.to_unit()
print(f'dirx = {dirx:.6f}\n')

diry = Vector(0.0, 1.0, 0.0)
print(f'diry = {diry:.6f}\n')

dirz = dirx.cross(diry)
print(f'dirz = {dirz:.6f}\n')

grids: list[list[ConstantGrid]] = []
k = 0
gridnose = ConstantGrid(k, *points[0, 0].to_xyz())
grids.append([gridnose for _ in range(points.shape[1])])
k += 1
for i in range(1, points.shape[0] - 1):
    grids.append([])
    for j in range(points.shape[1] - 1):
        grid = ConstantGrid(k, *points[i, j].to_xyz())
        # grid.ind = k
        grids[i].append(grid)
        k += 1
    grids[i].append(grids[i][0])
gridtail = ConstantGrid(k, *points[-1, 0].to_xyz())
grids.append([gridtail for _ in range(points.shape[1])])

ppoints = Vector.zeros((cnum - 1, bnum - 1))

panels: list[list[ConstantPanel]] = []

npanels: list[ConstantPanel] = []
wpanels: list[ConstantWakePanel] = []

dpanels: list[ConstantPanel] = []

k = 0
for i in range(cnum - 1):
    panels.append([])
    for j in range(bnum - 1):
        grida = grids[i + 1][j]
        gridb = grids[i][j]
        gridc = grids[i][j + 1]
        gridd = grids[i + 1][j + 1]
        if grida is gridb:
            dpanel = ConstantPanel(k, grida, gridc, gridd)
        elif gridb is gridc:
            dpanel = ConstantPanel(k, grida, gridb, gridd)
        elif gridc is gridd:
            dpanel = ConstantPanel(k, grida, gridb, gridc)
        elif gridd is grida:
            dpanel = ConstantPanel(k, gridb, gridc, gridd)
        else:
            dpanel = ConstantPanel(k, grida, gridb, gridc, gridd)
        k += 1
        dpanels.append(dpanel)
        panels[i].append(dpanel)
        ppoints[i, j] = dpanel.point
        # print(f'dpanel = {dpanel}')
        # print(f'dpanel.normal = {dpanel.normal}')
        # print(f'dpanel.point = {dpanel.point}')
        # print()
    # grida = grids[i][-1]
    # gridb = grids[i + 1][-1]
    # wpanel = ConstantWakePanel(k, grida, gridb, dirw)
    # k += 1
    # wpanels.append(wpanel)
    # print(f'wpanel = {wpanel}')
    # print(f'wpanel.normal = {wpanel.normal}')
    # print(f'wpanel.point = {wpanel.point}')
    # print()

sysname = 'Simple System'
sys = ConstantSystem(sysname, dpanels, npanels, wpanels)

sig = sys.unsig[:, 0].dot(vfs)
print(f'sig = {sig}\n')
print(f'sig.max() = {sig.max()}')
print(f'sig.min() = {sig.min()}')

mud = sys.unmud[:, 0].dot(vfs)
print(f'mud = {mud}\n')
print(f'mud.max() = {mud.max()}')
print(f'mud.min() = {mud.min()}')

mun = sys.unmun[:, 0].dot(vfs)
print(f'mun = {mun}\n')

muw = sys.unmuw[:, 0].dot(vfs)
print(f'muw = {muw}\n')

sys.grids

grid_index = zeros(points.shape, dtype=int)
for i in range(cnum):
    for j in range(bnum):
        grid = grids[i][j]
        grid_index[i, j] = grid.ind

panel_index = zeros(ppoints.shape, dtype=int)
for i in range(cnum - 1):
    for j in range(bnum - 1):
        panel = panels[i][j]
        panel_index[i, j] = panel.indo

phip = sys.amdd@mud + sys.amds@sig + sys.amdn@mun + sys.amdw@muw
print(f'phip = {phip}\n')

velp = sys.avdd@mud + sys.avds@sig + sys.avdn@mun + sys.avdw@muw + vfs
print(f'velp = {velp}\n')

velg = sys.avgd@mud + sys.avgs@sig + sys.avgn@mun + sys.avgw@muw + vfs
# print(f'velg = {velg}\n')

lvec = sys.blgd@mud + sys.blgn@mun + sys.blgw@muw
# print(f'lvec = \n{lvec}\n')

frcg = velg.cross(lvec)
# print(f'frcg = \n{frcg}\n')

prsg = frcg/sys.gridarea
# print(f'prsg = {prsg}\n')

gfrctot = frcg.sum()
print(f'gfrctot = {gfrctot:.6f}\n')

drag = gfrctot.dot(dirx)
print(f'drag = {drag:.6f}\n')

side = gfrctot.dot(diry)
print(f'side = {side:.6f}\n')

lift = gfrctot.dot(dirz)
print(f'lift = {lift:.6f}\n')

paramtable = MDParamTable()
paramtable.add_param('Drag Force', drag, '.6f', 'N')
paramtable.add_param('Side Force', side, '.6f', 'N')
paramtable.add_param('Lift Force', lift, '.6f', 'N')
display_markdown(paramtable)

#%%
# Create Plot
velg = sys.avgd@mud + sys.avgs@sig + sys.avgn@mun + sys.avgw@muw + vfs
# velg = sys.avgs@sig
# velg = sys.avgd@mud
# gvg = velg[grid_index]
velp = sys.avdd@mud + sys.avds@sig + sys.avdn@mun + sys.avdw@muw + vfs
# velp = sys.avds@sig
# velp = sys.avdd@mud

print(f'velg.x.max() = {velg.x.max()}')
print(f'velg.x.min() = {velg.x.min()}')
print(f'velg.y.max() = {velg.y.max()}')
print(f'velg.y.min() = {velg.y.min()}')
print(f'velg.z.max() = {velg.z.max()}')
print(f'velg.z.min() = {velg.z.min()}')

print()

print(f'velp.x.max() = {velp.x.max()}')
print(f'velp.x.min() = {velp.x.min()}')
print(f'velp.y.max() = {velp.y.max()}')
print(f'velp.y.min() = {velp.y.min()}')
print(f'velp.z.max() = {velp.z.max()}')
print(f'velp.z.min() = {velp.z.min()}')

scale = 0.1

unit_normals = normals.to_unit()

points = points.ravel()
ppoints = ppoints.ravel()
gprs = prsg[grid_index].ravel()/q
nprs = (unit_normals*prsg[grid_index].dot(unit_normals)).ravel()/q
gvel = velg[grid_index].ravel()*scale
vpr = velp.ravel()*scale

head_size = 0.5

gprsvecs = vectors(points.stack_xyz().astype('float32'),
                   gprs.stack_xyz().astype('float32'), color=0xff0000, head_size=head_size*scale)
nprsvecs = vectors(points.stack_xyz().astype('float32'),
                   nprs.stack_xyz().astype('float32'), color=0xff0000, head_size=head_size*scale)
gvelvecs = vectors(points.stack_xyz().astype('float32'),
                   gvel.stack_xyz().astype('float32'), color=0x00ff00, head_size=head_size*scale)
pvelvecs = vectors(ppoints.stack_xyz().astype('float32'),
                   vpr.stack_xyz().astype('float32'), color=0x0000ff, head_size=head_size*scale)

pnrmvecs = vectors(sys.dpoints.stack_xyz().astype('float32'),
                   sys.dnormal.stack_xyz().astype('float32'))

plot = Plot()
plot += k3d_surface(surface, unum=cnum, vnum=bnum, opacity=0.5)
# plot += nprsvecs
plot += gprsvecs
# plot += gvelvecs
# plot += pvelvecs
# plot += pnrmvecs
plot.display()

# #%%
# # Contour Plot
# num = 201
# zorg = 0.0
# pnts = point_mesh_xy(0.0, 0.0, zorg, num, num, 2.0, 2.0)

# fs = FreeStream(vfs)

# start = perf_counter()

# flw = fs.calculate_flow(pnts)

# for i, dpanel in enumerate(sys.dpanels):
#     sigi: float = sig[i]
#     mudi: float = mud[i]
#     flwdi, flwsi = dpanel.doublet_source_flow(pnts)
#     flw += flwdi*mudi
#     flw += flwsi*sigi

# finish = perf_counter()
# elapsed = finish - start
# print(f'Triangle time elapsed is {elapsed:.6f} seconds.')

# #%%
# # Doublet Velocity Potential
# levels = linspace(-2.0, 2.0, 21)
# axp, cfp = point_contourf_xy(pnts, flw.phi, levels=levels)
# _ = axp.set_title('3D Doublet Panel Velocity Potential - Triangle')

# # Doublet Velocity in X
# levels = linspace(0.0, 1.5, 21)
# axp, cfp = point_contourf_xy(pnts, flw.vel.x, levels=levels)
# _ = axp.set_title('3D Doublet Panel Velocity in X - Triangle')

# # Doublet Velocity in Y
# levels = linspace(-0.5, 0.5, 21)
# axp, cfp = point_contourf_xy(pnts, flw.vel.y, levels=levels)
# _ = axp.set_title('3D Doublet Panel Velocity in Y - Triangle')

# # Doublet Velocity in Z
# levels = linspace(-1.0, 1.0, 21)
# axp, cfp = point_contourf_xy(pnts, flw.vel.z, levels=levels)
# _ = axp.set_title('3D Doublet Panel Velocity in Z - Triangle')

#%%
