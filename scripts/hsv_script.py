#%%
# Import Dependencies
from time import perf_counter
from pygeom.geom3d import Vector
from pygeom.geom3d import zero_vector
from pyapm.classes.poly import Poly
from pyapm.classes.horseshoe import HorseShoe
from pyapm.tools.functions import mean, derivative
from matplotlib.pyplot import figure
from numpy import absolute

#%%
# Create Trailing Edge Vortex
grda = Vector(-0.25, 0.75, 0.0)
grdb = Vector(-1.25, -1.25, 0.0)
diro = Vector(1.0, 0.25, 0.0).to_unit()

hsv = HorseShoe(grda, grdb, diro)

grds = [
    grdb,
    grdb + 100*diro,
    grda + 100*diro,
    grda
]

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

pnts = zero_vector((numy, numx))
for i in range(numy):
    for j in range(numx):
        x = xorg-xamp+xint*j
        y = yorg-yamp+yint*i
        pnts[i, j] = Vector(x, y, zorg)

start = perf_counter()
phiv, velv = hsv.doublet_influence_coefficients(pnts)
phip, velp = poly.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

if zorg == 0.0:
    phiv[absolute(phiv) < 1e-12] = 0.0
    phiv[absolute(phiv-0.5) < 1e-12] = 0.5
    phiv[absolute(phiv+0.5) < 1e-12] = -0.5
    phip[absolute(phip) < 1e-12] = 0.0
    phip[absolute(phip-0.5) < 1e-12] = 0.5
    phip[absolute(phip+0.5) < 1e-12] = -0.5

#%%
# Doublet Panel Velocity Potential and Velocity in Z
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.x, pnts.y, phiv, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.x, pnts.y, phip, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.x, pnts.y, velv.z, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.x, pnts.y, velp.z, levels = 20)
cbv = figv.colorbar(csv)

#%%
# Doublet Velocity in Y
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Y')
csv = axv.contourf(pnts.x, pnts.y, velv.y, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Y')
csv = axv.contourf(pnts.x, pnts.y, velp.y, levels = 20)
cbv = figv.colorbar(csv)

velvy = derivative(phiv, pnts.y)
pntsx = mean(pnts.x)
pntsy = mean(pnts.y)

figs = figure(figsize = (12, 10))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in Y')
css = axs.contourf(pntsx, pntsy, velvy, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Doublet Velocity in X
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in X')
csv = axv.contourf(pnts.x, pnts.y, velv.x, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in X')
csv = axv.contourf(pnts.x, pnts.y, velp.x, levels = 20)
cbv = figv.colorbar(csv)

velvx = derivative(phiv, pnts.x, axis=1)
pntsx = mean(pnts.x, axis=1)
pntsy = mean(pnts.y, axis=1)

figs = figure(figsize = (12, 10))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in X')
css = axs.contourf(pntsx, pntsy, velvx, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Mesh Points
xorg = 0.0
yorg = -1.5
zorg = 0.0
numx = 201
numz = 201
xamp = 2.0
zamp = 2.0
xint = 2*xamp/(numx-1)
zint = 2*zamp/(numz-1)

pnts = zero_vector((numy, numx))
for i in range(numz):
    for j in range(numx):
        x = xorg-xamp+xint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(x, yorg, z)

start = perf_counter()
phiv, velv = hsv.doublet_influence_coefficients(pnts)
phip, velp = poly.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.x, pnts.z, phiv, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.x, pnts.z, phip, levels = 20)
cbv = figv.colorbar(csv)

#%%
# Doublet Velocity in Z
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.x, pnts.z, velv.z, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.x, pnts.z, velp.z, levels = 20)
cbv = figv.colorbar(csv)

velvz = derivative(phiv, pnts.z)
pntsx = mean(pnts.x)
pntsz = mean(pnts.z)

figs = figure(figsize = (12, 10))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in Z')
css = axs.contourf(pntsx, pntsz, velvz, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Doublet Velocity in X
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in X')
csv = axv.contourf(pnts.x, pnts.z, velv.x, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in X')
csv = axv.contourf(pnts.x, pnts.z, velp.x, levels = 20)
cbv = figv.colorbar(csv)

velvx = derivative(phiv, pnts.x, axis=1)
pntsx = mean(pnts.x, axis=1)
pntsz = mean(pnts.z, axis=1)

figs = figure(figsize = (12, 10))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in X')
css = axs.contourf(pntsx, pntsz, velvx, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Mesh Points
xorg = 0.0
yorg = 0.0
zorg = 0.0
numy = 201
numz = 201
yamp = 2.0
zamp = 2.0
yint = 2*yamp/(numy-1)
zint = 2*zamp/(numz-1)

pnts = zero_vector((numy, numy))
for i in range(numz):
    for j in range(numy):
        y = yorg-yamp+yint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(xorg, y, z)

start = perf_counter()
phiv, velv = hsv.doublet_influence_coefficients(pnts)
phip, velp = poly.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.y, pnts.z, phiv, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity Potential')
csv = axv.contourf(pnts.y, pnts.z, phip, levels = 20)
cbv = figv.colorbar(csv)

#%%
# Doublet Velocity in Z
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.y, pnts.z, velv.z, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Z')
csv = axv.contourf(pnts.y, pnts.z, velp.z, levels = 20)
cbv = figv.colorbar(csv)

velvz = derivative(phiv, pnts.z)
pntsy = mean(pnts.y)
pntsz = mean(pnts.z)

figs = figure(figsize = (12, 10))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in Z')
css = axs.contourf(pntsy, pntsz, velvz, levels = csv.levels)
cbs = figs.colorbar(css)

#%%
# Doublet Velocity in X
figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Y')
csv = axv.contourf(pnts.y, pnts.z, velv.y, levels = 20)
cbv = figv.colorbar(csv)

figv = figure(figsize = (12, 10))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('3D Doublet Panel Velocity in Y')
csv = axv.contourf(pnts.y, pnts.z, velp.y, levels = 20)
cbv = figv.colorbar(csv)

velvy = derivative(phiv, pnts.y, axis=1)
pntsy = mean(pnts.y, axis=1)
pntsz = mean(pnts.z, axis=1)

figs = figure(figsize = (12, 10))
axs = figs.gca()
axs.set_aspect('equal')
axs.set_title('3D Doublet Panel Velocity in Y')
css = axs.contourf(pntsy, pntsz, velvy, levels = csv.levels)
cbs = figs.colorbar(css)
