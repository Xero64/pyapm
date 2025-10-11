#%%
# Import Dependencies
from time import perf_counter

from numpy.linalg import norm
from pyapm.tools.cupy import cupy_cwdf, cupy_cwdp, cupy_cwdv
from pyapm.tools.mesh import point_mesh_xy, point_mesh_yz, point_mesh_zx
from pyapm.tools.numpy import numpy_cwdf, numpy_cwdp, numpy_cwdv
from pyapm.tools.plot import (point_contourf_xy, point_contourf_xz,
                               point_contourf_yz)
from pygeom.geom3d import Vector

#%%
# Create Trailing Doublet
num = 201

# Trailing Doublet
grda = Vector(-1.2, 1.0, 0.0)
grdb = grda
dirw = Vector(1.0, -0.25, 0.0).to_unit()

cond = -1.0

#%%
# Create Inputs
zorg = 0.0
pnts = point_mesh_xy(0.0, 0.0, zorg, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy total time elapsed is {elapsedc:.6f} seconds.')
print(f'Speedup is {elapsedn / elapsedc:.2f}x.')

#%%
# Doublet in XY
diffpho = flwn.phi - flwc.phi
diffvxo = flwn.vel.x - flwc.vel.x
diffvyo = flwn.vel.y - flwc.vel.y
diffvzo = flwn.vel.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

# Doublet Velocity Potential
axd, cfd = point_contourf_xy(pnts, flwn.phi)
_ = axd.set_title('3D Doublet Panel Velocity Potential - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.phi)
_ = axp.set_title('3D Doublet Panel Velocity Potential - Cupy')

# Doublet Velocity in X
axd, cfd = point_contourf_xy(pnts, flwn.vel.x)
_ = axd.set_title('3D Doublet Panel Velocity in X - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.x)
_ = axp.set_title('3D Doublet Panel Velocity in X - Cupy')

# Doublet Velocity in Y
axd, cfd = point_contourf_xy(pnts, flwn.vel.y)
_ = axd.set_title('3D Doublet Panel Velocity in Y - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.y)
_ = axp.set_title('3D Doublet Panel Velocity in Y - Cupy')

# Doublet Velocity in Z
axd, cfd = point_contourf_xy(pnts, flwn.vel.z)
_ = axd.set_title('3D Doublet Panel Velocity in Z - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.z)
_ = axp.set_title('3D Doublet Panel Velocity in Z - Cupy')

#%%
# Check Components
phic = cupy_cwdp(pnts, grda, grdb, dirw, cond = cond)
velc = cupy_cwdv(pnts, grda, grdb, dirw)

diffpho = phic - flwc.phi
diffvxo = velc.x - flwc.vel.x
diffvyo = velc.y - flwc.vel.y
diffvzo = velc.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

phin = numpy_cwdp(pnts, grda, grdb, dirw, cond = cond)
veln = numpy_cwdv(pnts, grda, grdb, dirw)

diffpho = phin - flwn.phi
diffvxo = veln.x - flwn.vel.x
diffvyo = veln.y - flwn.vel.y
diffvzo = veln.z - flwn.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

#%%
# Mesh Points
zorg = -0.05
pnts = point_mesh_xy(0.0, 0.0, zorg, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy total time elapsed is {elapsedc:.6f} seconds.')
print(f'Speedup is {elapsedn / elapsedc:.2f}x.')

#%%
# Doublet in XY
diffpho = flwn.phi - flwc.phi
diffvxo = flwn.vel.x - flwc.vel.x
diffvyo = flwn.vel.y - flwc.vel.y
diffvzo = flwn.vel.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

# Doublet Velocity Potential
axd, cfd = point_contourf_xy(pnts, flwn.phi)
_ = axd.set_title('3D Doublet Panel Velocity Potential - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.phi)
_ = axp.set_title('3D Doublet Panel Velocity Potential - Cupy')

# Doublet Velocity in X
axd, cfd = point_contourf_xy(pnts, flwn.vel.x)
_ = axd.set_title('3D Doublet Panel Velocity in X - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.x)
_ = axp.set_title('3D Doublet Panel Velocity in X - Cupy')

# Doublet Velocity in Y
axd, cfd = point_contourf_xy(pnts, flwn.vel.y)
_ = axd.set_title('3D Doublet Panel Velocity in Y - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.y)
_ = axp.set_title('3D Doublet Panel Velocity in Y - Cupy')

# Doublet Velocity in Z
axd, cfd = point_contourf_xy(pnts, flwn.vel.z)
_ = axd.set_title('3D Doublet Panel Velocity in Z - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.z)
_ = axp.set_title('3D Doublet Panel Velocity in Z - Cupy')

#%%
# Check Components
phic = cupy_cwdp(pnts, grda, grdb, dirw, cond = cond)
velc = cupy_cwdv(pnts, grda, grdb, dirw)

diffpho = phic - flwc.phi
diffvxo = velc.x - flwc.vel.x
diffvyo = velc.y - flwc.vel.y
diffvzo = velc.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

phin = numpy_cwdp(pnts, grda, grdb, dirw, cond = cond)
veln = numpy_cwdv(pnts, grda, grdb, dirw)

diffpho = phin - flwn.phi
diffvxo = veln.x - flwn.vel.x
diffvyo = veln.y - flwn.vel.y
diffvzo = veln.z - flwn.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

#%%
# Mesh Points
yorg = -0.05
pnts = point_mesh_zx(0.0, yorg, 0.0, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy total time elapsed is {elapsedc:.6f} seconds.')
print(f'Speedup is {elapsedn / elapsedc:.2f}x.')

#%%
# Doublet in XZ
diffpho = flwn.phi - flwc.phi
diffvxo = flwn.vel.x - flwc.vel.x
diffvyo = flwn.vel.y - flwc.vel.y
diffvzo = flwn.vel.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

# Doublet Velocity Potential
axd, cfd = point_contourf_xz(pnts, flwn.phi)
_ = axd.set_title('3D Doublet Panel Velocity Potential - Numpy')

axp, cfp = point_contourf_xz(pnts, flwc.phi)
_ = axp.set_title('3D Doublet Panel Velocity Potential - Cupy')

# Doublet Velocity Velocity in X
axd, cfd = point_contourf_xz(pnts, flwn.vel.x)
_ = axd.set_title('3D Doublet Panel Velocity in X - Numpy')

axp, cfp = point_contourf_xz(pnts, flwc.vel.x)
_ = axp.set_title('3D Doublet Panel Velocity in X - Cupy')

# Doublet Velocity Velocity in Y
axd, cfd = point_contourf_xz(pnts, flwn.vel.y)
_ = axd.set_title('3D Doublet Panel Velocity in Y - Numpy')

axp, cfp = point_contourf_xz(pnts, flwc.vel.y)
_ = axp.set_title('3D Doublet Panel Velocity in Y - Cupy')

# Doublet Velocity Velocity in Z
axd, cfd = point_contourf_xz(pnts, flwn.vel.z)
_ = axd.set_title('3D Doublet Panel Velocity in Z - Numpy')

axp, cfp = point_contourf_xz(pnts, flwc.vel.z)
_ = axp.set_title('3D Doublet Panel Velocity in Z - Cupy')

#%%
# Check Components
phic = cupy_cwdp(pnts, grda, grdb, dirw, cond = cond)
velc = cupy_cwdv(pnts, grda, grdb, dirw)

diffpho = phic - flwc.phi
diffvxo = velc.x - flwc.vel.x
diffvyo = velc.y - flwc.vel.y
diffvzo = velc.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

phin = numpy_cwdp(pnts, grda, grdb, dirw, cond = cond)
veln = numpy_cwdv(pnts, grda, grdb, dirw)

diffpho = phin - flwn.phi
diffvxo = veln.x - flwn.vel.x
diffvyo = veln.y - flwn.vel.y
diffvzo = veln.z - flwn.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

#%%
# Mesh Points
xorg = -0.05
pnts = point_mesh_yz(xorg, 0.0, 0.0, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy total time elapsed is {elapsedc:.6f} seconds.')
print(f'Speedup is {elapsedn / elapsedc:.2f}x.')

#%%
# Doublet in YZ
diffpho = flwn.phi - flwc.phi
diffvxo = flwn.vel.x - flwc.vel.x
diffvyo = flwn.vel.y - flwc.vel.y
diffvzo = flwn.vel.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

# Doublet Velocity Potential
axd, cfd = point_contourf_yz(pnts, flwn.phi)
_ = axd.set_title('3D Doublet Panel Velocity Potential - Numpy')

axp, cfp = point_contourf_yz(pnts, flwc.phi)
_ = axp.set_title('3D Doublet Panel Velocity Potential - Cupy')

# Doublet Velocity Velocity in X
axd, cfd = point_contourf_yz(pnts, flwn.vel.x)
_ = axd.set_title('3D Doublet Panel Velocity in X - Numpy')

axp, cfp = point_contourf_yz(pnts, flwc.vel.x)
_ = axp.set_title('3D Doublet Panel Velocity in X - Cupy')

# Doublet Velocity Velocity in Y
axd, cfd = point_contourf_yz(pnts, flwn.vel.y)
_ = axd.set_title('3D Doublet Panel Velocity in Y - Numpy')

axp, cfp = point_contourf_yz(pnts, flwc.vel.y)
_ = axp.set_title('3D Doublet Panel Velocity in Y - Cupy')

# Doublet Velocity Velocity in Z
axd, cfd = point_contourf_yz(pnts, flwn.vel.z)
_ = axd.set_title('3D Doublet Panel Velocity in Z - Numpy')

axp, cfp = point_contourf_yz(pnts, flwc.vel.z)
_ = axp.set_title('3D Doublet Panel Velocity in Z - Cupy')

#%%
# Check Components
phic = cupy_cwdp(pnts, grda, grdb, dirw, cond = cond)
velc = cupy_cwdv(pnts, grda, grdb, dirw)

diffpho = phic - flwc.phi
diffvxo = velc.x - flwc.vel.x
diffvyo = velc.y - flwc.vel.y
diffvzo = velc.z - flwc.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

phin = numpy_cwdp(pnts, grda, grdb, dirw, cond = cond)
veln = numpy_cwdv(pnts, grda, grdb, dirw)

diffpho = phin - flwn.phi
diffvxo = veln.x - flwn.vel.x
diffvyo = veln.y - flwn.vel.y
diffvzo = veln.z - flwn.vel.z

normpho = norm(diffpho)
normvxo = norm(diffvxo)
normvyo = norm(diffvyo)
normvzo = norm(diffvzo)

print(f'normpho = {normpho:.12f}')
print(f'normvxo = {normvxo:.12f}')
print(f'normvyo = {normvyo:.12f}')
print(f'normvzo = {normvzo:.12f}')

#%%
# Corner Point A
pnts = grda

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Corner Point B
pnts = grdb

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Centre Point
pnts = (grda + grdb)/2.0 + dirw

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Edge Point
pnts = (grda + grdb)/2.0

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Outside Point
pnts = (grda + grdb)/2.0 + dirw

start = perf_counter()
flwn = numpy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_cwdf(pnts, grda, grdb, dirw, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')
