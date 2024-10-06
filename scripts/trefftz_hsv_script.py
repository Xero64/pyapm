#%%
# Import Dependencies
from time import perf_counter

from matplotlib.pyplot import figure

from pyapm.classes.grid import Grid
from pyapm.classes.horseshoe import HorseShoe
from pygeom.geom3d import Vector

#%%
# Create Horseshoe Vortex
grda = Grid(1, 0.0, -1.0, 0.0)
grdb = Grid(2, 0.0, 1.0, 0.0)
diro = Vector(1.0, 0.0, 0.0).to_unit()

hsv = HorseShoe(grda, grdb, diro)

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

pnts = Vector.zeros((numz, numy))
for i in range(numz):
    for j in range(numy):
        y = yorg-yamp+yint*j
        z = zorg-zamp+zint*i
        pnts[i, j] = Vector(xorg, y, z)

start = perf_counter()
vel = hsv.trefftz_plane_velocities(pnts)
finished = perf_counter()
elapsed = finished-start
print(f'Time elapsed is {elapsed:.2f} seconds.')

#%%
# Horseshoe Vortex Velocity in X
figv = figure(figsize = (12, 12))
axv = figv.gca()
axv.set_aspect('equal')
axv.set_title('Horseshoe Vortex Velocity in X')
csv = axv.contourf(pnts.y, pnts.z, vel.x, levels = 20)
cbv = figv.colorbar(csv)

#%%
# Horseshoe Vortex Velocity in Y
figs = figure(figsize = (12, 12))
axv = figs.gca()
axv.set_aspect('equal')
axv.set_title('Horseshoe Vortex Velocity in Y')
csv = axv.contourf(pnts.y, pnts.z, vel.y, levels = 20)
cbv = figs.colorbar(csv)

#%%
# Horseshoe Vortex Velocity in Z
figs = figure(figsize = (12, 12))
axv = figs.gca()
axv.set_aspect('equal')
axv.set_title('Horseshoe Vortex Velocity in Z')
csv = axv.contourf(pnts.y, pnts.z, vel.z, levels = 20)
cbv = figs.colorbar(csv)
