#%%
# Import Dependencies
from time import perf_counter

from matplotlib.pyplot import figure
from numpy import absolute
from pyapm.classes.horseshoe import HorseShoe
from pyapm.classes.horseshoedoublet import HorseshoeDoublet
from pygeom.geom3d import Vector

#%%
# Create Trailing Edge Vortex
# grda = Vector(-0.25, 0.75, 0.0)
# grdb = Vector(-1.25, -1.25, 0.0)
# diro = Vector(1.0, 0.25, 0.0).to_unit()
grda = Vector(-1.0, 1.0, 0.0)
grdb = Vector(-1.0, -1.0, 0.0)
diro = Vector(1.0, 0.0, 0.0).to_unit()

hsv = HorseShoe(grda, grdb, diro)
hsd = HorseshoeDoublet(grda, grdb, diro)

#%%
# Mesh Points
xorg = 0.0
yorg = 0.0
zorg = 0.0
numx = 201
numy = 201
xamp = 2.0
yamp = 2.0
xint = 2*xamp/(numx-1)
yint = 2*yamp/(numy-1)

pnts = Vector.zeros((numy, numx))
for i in range(numy):
    for j in range(numx):
        x = xorg-xamp+xint*j
        y = yorg-yamp+yint*i
        pnts[i, j] = Vector(x, y, zorg)

start = perf_counter()
phiv, velv = hsv.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Horseshoe Time elapsed is {elapsed:.2f} seconds.')

start = perf_counter()
phid, veld = hsd.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Horseshoe Doublet Time elapsed is {elapsed:.2f} seconds.')

if zorg == 0.0:
    phiv[absolute(phiv) < 1e-12] = 0.0
    phiv[absolute(phiv-0.5) < 1e-12] = 0.5
    phiv[absolute(phiv+0.5) < 1e-12] = -0.5
    phid[absolute(phid) < 1e-12] = 0.0
    phid[absolute(phid-0.5) < 1e-12] = 0.5
    phid[absolute(phid+0.5) < 1e-12] = -0.5

#%%
# Doublet Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity Potential - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.y, phiv, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity Potential - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.y, phid, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in X - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.y, velv.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in X - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.y, veld.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Y - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.y, velv.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Y - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.y, veld.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Z - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.y, velv.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Z - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.y, veld.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)

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

pnts = Vector.zeros((numy, numx))
for i in range(numz):
    for j in range(numx):
        x = xorg-xamp+xint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(x, yorg, z)

start = perf_counter()
phiv, velv = hsv.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Horseshoe Time elapsed is {elapsed:.2f} seconds.')

start = perf_counter()
phid, veld = hsd.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Horseshoe Doublet Time elapsed is {elapsed:.2f} seconds.')

#%%
# Doublet Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity Potential - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.z, phiv, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity Potential - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.z, phid, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in X - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.z, velv.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in X - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.z, veld.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Y - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.z, velv.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Y - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.z, veld.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Z - Horseshoe')
cfp = axp.contourf(pnts.x, pnts.z, velv.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Z - Horseshoe Doublet')
cfd = axd.contourf(pnts.x, pnts.z, veld.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)

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

pnts = Vector.zeros((numy, numy))
for i in range(numz):
    for j in range(numy):
        y = yorg-yamp+yint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(xorg, y, z)

start = perf_counter()
phiv, velv = hsv.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Horseshoe Time elapsed is {elapsed:.2f} seconds.')

start = perf_counter()
phid, veld = hsd.doublet_influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Horseshoe Doublet Time elapsed is {elapsed:.2f} seconds.')

#%%
# Doublet Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity Potential - Horseshoe')
cfp = axp.contourf(pnts.y, pnts.z, phiv, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity Potential - Horseshoe Doublet')
cfd = axd.contourf(pnts.y, pnts.z, phid, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in X - Horseshoe')
cfp = axp.contourf(pnts.y, pnts.z, velv.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in X - Horseshoe Doublet')
cfd = axd.contourf(pnts.y, pnts.z, veld.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Y - Horseshoe')
cfp = axp.contourf(pnts.y, pnts.z, velv.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Y - Horseshoe Doublet')
cfd = axd.contourf(pnts.y, pnts.z, veld.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Z - Horseshoe')
cfp = axp.contourf(pnts.y, pnts.z, velv.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Z - Horseshoe Doublet')
cfd = axd.contourf(pnts.y, pnts.z, veld.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)
