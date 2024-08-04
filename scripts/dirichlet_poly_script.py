#%%
# Import Dependencies
from time import perf_counter
# from math import pi, cos, sin
from pygeom.geom3d import Vector
from pygeom.array3d import zero_arrayvector
from pyapm.classes.poly import Poly
from pyapm.classes.dirichletpoly import DirichletPoly
from matplotlib.pyplot import figure
from numpy import absolute

#%%
# Create Poly

# # Star
# num = 10
# thint = 2*pi/num
# ro = 1.5
# ri = 0.5
# th = [thint/2 + i*thint for i in range(num)]

# grds = []
# for i in range(num):
#     if i % 2 == 0:
#         grds.append(Vector(ro*cos(th[i]), ro*sin(th[i]), 0.0))
#     else:
#         grds.append(Vector(ri*cos(th[i]), ri*sin(th[i]), 0.0))

# Quadrilateral
grds = [
    Vector(-1.0, -1.0, 0.0),
    Vector(1.0, -1.0, 0.0),
    Vector(0.0, 1.0, 0.0)
]

#%%
# Create Poly
poly = Poly(grds)

#%%
# Create Dirichlet Poly
dpnl = DirichletPoly(grds)

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

pnts = zero_arrayvector((numy, numx))
for i in range(numy):
    for j in range(numx):
        x = xorg-xamp+xint*j
        y = yorg-yamp+yint*i
        pnts[i, j] = Vector(x, y, zorg)

start = perf_counter()
phid, phis, veld, vels = dpnl.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Dirichlet Panel time elapsed is {elapsed:.6f} seconds.')

start = perf_counter()
phidp, phisp, veldp, velsp = poly.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Poly time elapsed is {elapsed:.6f} seconds.')

if zorg == 0.0:
    phid[absolute(phid) < 1e-12] = 0.0
    phid[absolute(phid-0.5) < 1e-12] = 0.5
    phid[absolute(phid+0.5) < 1e-12] = -0.5
    vels.z[absolute(vels.z) < 1e-12] = 0.0
    vels.z[absolute(vels.z-0.5) < 1e-12] = 0.5
    vels.z[absolute(vels.z+0.5) < 1e-12] = -0.5
    phidp[absolute(phidp) < 1e-12] = 0.0
    phidp[absolute(phidp-0.5) < 1e-12] = 0.5
    phidp[absolute(phidp+0.5) < 1e-12] = -0.5
    velsp.z[absolute(velsp.z) < 1e-12] = 0.0
    velsp.z[absolute(velsp.z-0.5) < 1e-12] = 0.5
    velsp.z[absolute(velsp.z+0.5) < 1e-12] = -0.5

#%%
# Doublet Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity Potential - Poly')
cfp = axp.contourf(pnts.x, pnts.y, phidp, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity Potential - Dirichlet Panel')
# cfd = axd.contourf(pnts.x, pnts.y, phid, levels = cfp.levels)
cfd = axd.contourf(pnts.x, pnts.y, phid, levels = 20)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in X - Poly')
cfp = axp.contourf(pnts.x, pnts.y, veldp.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in X - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.y, veld.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Y - Poly')
cfp = axp.contourf(pnts.x, pnts.y, veldp.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Y - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.y, veld.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Z - Poly')
cfp = axp.contourf(pnts.x, pnts.y, veldp.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Z - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.y, veld.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity Potential - Poly')
cfp = axp.contourf(pnts.x, pnts.y, phisp, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity Potential - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.y, phis, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in X - Poly')
cfp = axp.contourf(pnts.x, pnts.y, velsp.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in X - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.y, vels.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in Y - Poly')
cfp = axp.contourf(pnts.x, pnts.y, velsp.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in Y - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.y, vels.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in Z - Poly')
cfp = axp.contourf(pnts.x, pnts.y, velsp.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in Z - Dirichlet Panel')
# cfd = axd.contourf(pnts.x, pnts.y, vels.z, levels = cfp.levels)
cfd = axd.contourf(pnts.x, pnts.y, vels.z, levels = 20)
cbd = figd.colorbar(cfd)

#%%
# Mesh Points
xorg = 0.0
yorg = -0.05
zorg = 0.0
numx = 201
numz = 201
xamp = 2.0
zamp = 2.0
xint = 2*xamp/(numx-1)
zint = 2*zamp/(numz-1)

pnts = zero_arrayvector((numy, numx))
for i in range(numz):
    for j in range(numx):
        x = xorg-xamp+xint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(x, yorg, z)

start = perf_counter()
phid, phis, veld, vels = dpnl.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Dirichlet Panel time elapsed is {elapsed:.6f} seconds.')

start = perf_counter()
phidp, phisp, veldp, velsp = poly.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Poly time elapsed is {elapsed:.6f} seconds.')

#%%
# Doublet Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity Potential - Poly')
cfp = axp.contourf(pnts.x, pnts.z, phidp, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity Potential - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, phid, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in X - Poly')
cfp = axp.contourf(pnts.x, pnts.z, veldp.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in X - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, veld.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Y - Poly')
cfp = axp.contourf(pnts.x, pnts.z, veldp.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Y - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, veld.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Z - Poly')
cfp = axp.contourf(pnts.x, pnts.z, veldp.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Z - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, veld.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity Potential - Poly')
cfp = axp.contourf(pnts.x, pnts.z, phisp, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity Potential - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, phis, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in X - Poly')
cfp = axp.contourf(pnts.x, pnts.z, velsp.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in X - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, vels.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in Y - Poly')
cfp = axp.contourf(pnts.x, pnts.z, velsp.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in Y - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, vels.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in Z - Poly')
cfp = axp.contourf(pnts.x, pnts.z, velsp.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in Z - Dirichlet Panel')
cfd = axd.contourf(pnts.x, pnts.z, vels.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Mesh Points
xorg = -0.05
yorg = 0.0
zorg = 0.0
numy = 201
numz = 201
yamp = 2.0
zamp = 2.0
yint = 2*yamp/(numy-1)
zint = 2*zamp/(numz-1)

pnts = zero_arrayvector((numy, numy))
for i in range(numz):
    for j in range(numy):
        y = yorg-yamp+yint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(xorg, y, z)

start = perf_counter()
phid, phis, veld, vels = dpnl.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Dirichlet Panel time elapsed is {elapsed:.6f} seconds.')

start = perf_counter()
phidp, phisp, veldp, velsp = poly.influence_coefficients(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Poly time elapsed is {elapsed:.6f} seconds.')

#%%
# Doublet Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity Potential - Poly')
cfp = axp.contourf(pnts.y, pnts.z, phidp, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity Potential - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, phid, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in X - Poly')
cfp = axp.contourf(pnts.y, pnts.z, veldp.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in X - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, veld.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Y - Poly')
cfp = axp.contourf(pnts.y, pnts.z, veldp.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Y - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, veld.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Doublet Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Doublet Panel Velocity in Z - Poly')
cfp = axp.contourf(pnts.y, pnts.z, veldp.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Doublet Panel Velocity in Z - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, veld.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity Potential
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity Potential - Poly')
cfp = axp.contourf(pnts.y, pnts.z, phisp, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity Potential - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, phis, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in X
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in X - Poly')
cfp = axp.contourf(pnts.y, pnts.z, velsp.x, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in X - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, vels.x, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in Y
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in Y - Poly')
cfp = axp.contourf(pnts.y, pnts.z, velsp.y, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in Y - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, vels.y, levels = cfp.levels)
cbd = figd.colorbar(cfd)

#%%
# Source Velocity in Z
figp = figure(figsize = (12, 10))
axp = figp.gca()
axp.set_aspect('equal')
axp.set_title('3D Source Panel Velocity in Z - Poly')
cfp = axp.contourf(pnts.y, pnts.z, velsp.z, levels = 20)
cbp = figp.colorbar(cfp)

figd = figure(figsize = (12, 10))
axd = figd.gca()
axd.set_aspect('equal')
axd.set_title('3D Source Panel Velocity in Z - Dirichlet Panel')
cfd = axd.contourf(pnts.y, pnts.z, vels.z, levels = cfp.levels)
cbd = figd.colorbar(cfd)
