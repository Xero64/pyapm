#%%
# Import Dependencies
from time import perf_counter
from math import pi, cos, sin
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector
from pyapm.classes.poly import Poly
from pyapm.tools.functions import mean, derivative
from matplotlib.pyplot import figure
from numpy.matlib import absolute

#%%
# Create Poly

# Star
num = 10
thint = 2*pi/num
ro = 1.5
ri = 0.5
th = [thint/2 + i*thint for i in range(num)]

grds = []
for i in range(num):
    if i % 2 == 0:
        grds.append(Vector(ro*cos(th[i]), ro*sin(th[i]), 0.0))
    else:
        grds.append(Vector(ri*cos(th[i]), ri*sin(th[i]), 0.0))

# # Quadrilateral
# grds = [
#     Vector(-1.0, -1.0, 0.0),
#     Vector(1.0, -1.0, 0.0),
#     Vector(1.0, 1.0, 0.0),
#     Vector(-1.0, 1.0, 0.0)
# ]

#%%
# Print Bound Edge
poly = Poly(grds)

#%%
# Mesh Points
xorg = 0.0
yorg = 0.0
zorg = -0.05
numx = 201
numy = 201
xamp = 2.0
yamp = 2.0
xint = 2*xamp/(numx-1)
yint = 2*yamp/(numy-1)

pnts = zero_matrix_vector((numy, numx))
for i in range(numy):
    for j in range(numx):
        x = xorg-xamp+xint*j
        y = yorg-yamp+yint*i
        pnts[i, j] = Vector(x, y, zorg)

start = perf_counter()
phiv, phis, velv, vels = poly.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

start = perf_counter()
phiv, phis = poly.velocity_potentials(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

if zorg == 0.0:
    phiv[absolute(phiv) < 1e-12] = 0.0
    phiv[absolute(phiv-0.5) < 1e-12] = 0.5
    phiv[absolute(phiv+0.5) < 1e-12] = -0.5
    vels.z[absolute(vels.z) < 1e-12] = 0.0
    vels.z[absolute(vels.z-0.5) < 1e-12] = 0.5
    vels.z[absolute(vels.z+0.5) < 1e-12] = -0.5

#%%
# Source Panel Velocity Potential and Velocity in Z
figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity Potential')
css = axs.contourf(pnts.x, pnts.y, phis, levels = 20)
cbs = figs.colorbar(css)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in Z')
css = axs.contourf(pnts.x, pnts.y, vels.z, levels = 20)
cbs = figs.colorbar(css)

#%%
# Doublet Panel Velocity Potential and Velocity in Z
figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.x, pnts.y, phiv, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.x, pnts.y, velv.z, levels = 20)
cbv = figv.colorbar(csv)

#%%
# Doublet Velocity in Y
figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Y')
csv = axv.contourf(pnts.x, pnts.y, velv.y, levels = 20)
cbv = figv.colorbar(csv)

velvy = derivative(phiv, pnts.y)
pntsx = mean(pnts.x)
pntsy = mean(pnts.y)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in Y')
css = axs.contourf(pntsx, pntsy, velvy, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Doublet Velocity in X
figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in X')
csv = axv.contourf(pnts.x, pnts.y, velv.x, levels = 20)
cbv = figv.colorbar(csv)

velvx = derivative(phiv, pnts.x, axis=1)
pntsx = mean(pnts.x, axis=1)
pntsy = mean(pnts.y, axis=1)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in X')
css = axs.contourf(pntsx, pntsy, velvx, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Source Velocity in Y
figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in Y')
css = axs.contourf(pnts.x, pnts.y, vels.y, levels = 20)
cbs = figs.colorbar(css)

velsy = derivative(phis, pnts.y)
pntsx = mean(pnts.x)
pntsy = mean(pnts.y)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in Y')
css = axs.contourf(pntsx, pntsy, velsy, levels = css.levels)
cbs = figs.colorbar(css)

#%%
# Source Velocity in X
figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in X')
css = axs.contourf(pnts.x, pnts.y, vels.x, levels = 20)
cbs = figs.colorbar(css)

velsx = derivative(phis, pnts.x, axis=1)
pntsx = mean(pnts.x, axis=1)
pntsy = mean(pnts.y, axis=1)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in X')
css = axs.contourf(pntsx, pntsy, velsx, levels = css.levels)
cbs = figs.colorbar(css)

#%%
# Mesh Points
xorg = 0.0
yorg = 0.0
zorg = 0.0
numx = 201
numz = 201
xamp = 2.0
zamp = 2.0
xint = 2*xamp/(numx-1)
zint = 2*zamp/(numz-1)

pnts = zero_matrix_vector((numy, numx))
for i in range(numz):
    for j in range(numx):
        x = xorg-xamp+xint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(x, yorg, z)

start = perf_counter()
phiv, phis, velv, vels = poly.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.x, pnts.z, phiv, levels = 20)
cbv = figv.colorbar(csv)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity Potential')
css = axs.contourf(pnts.x, pnts.z, phis, levels = 20)
cbs = figs.colorbar(css)

#%%
# Doublet Velocity in Z
figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.x, pnts.z, velv.z, levels = 20)
cbv = figv.colorbar(csv)

velvz = derivative(phiv, pnts.z)
pntsx = mean(pnts.x)
pntsz = mean(pnts.z)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in Z')
css = axs.contourf(pntsx, pntsz, velvz, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Doublet Velocity in X
figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in X')
csv = axv.contourf(pnts.x, pnts.z, velv.x, levels = 20)
cbv = figv.colorbar(csv)

velvx = derivative(phiv, pnts.x, axis=1)
pntsx = mean(pnts.x, axis=1)
pntsz = mean(pnts.z, axis=1)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in X')
css = axs.contourf(pntsx, pntsz, velvx, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Source Velocity in Z
figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in Z')
css = axs.contourf(pnts.x, pnts.z, vels.z, levels = 20)
cbs = figs.colorbar(css)

velsz = derivative(phis, pnts.z)
pntsx = mean(pnts.x)
pntsz = mean(pnts.z)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in Z')
css = axs.contourf(pntsx, pntsz, velsz, levels = css.levels)
cbs = figs.colorbar(css)

#%%
# Source Velocity in X
figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in X')
css = axs.contourf(pnts.x, pnts.z, vels.x, levels = 20)
cbs = figs.colorbar(css)

velsx = derivative(phis, pnts.x, axis=1)
pntsx = mean(pnts.x, axis=1)
pntsz = mean(pnts.z, axis=1)

figs = figure(figsize = (12, 12))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Source Panel Velocity in X')
css = axs.contourf(pntsx, pntsz, velsx, levels = css.levels)
cbs = figs.colorbar(css)
