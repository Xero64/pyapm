#%%
# Import Dependencies
from time import perf_counter

from numpy import cos, radians
from numpy.linalg import norm
from pyapm.tools.cupy import cupy_ctdf, cupy_ctdp, cupy_ctdv
from pyapm.tools.mesh import point_mesh_xy, point_mesh_yz, point_mesh_zx
from pyapm.tools.numpy import numpy_ctdf, numpy_ctdp, numpy_ctdv
from pyapm.tools.plot import (point_contourf_xy, point_contourf_xz,
                               point_contourf_yz)
from pygeom.geom3d import Vector

#%%
# Create Triangle
num = 201

rad30 = radians(30.0)
cos30 = cos(rad30)
sin30 = 0.5

grda = Vector(-cos30, -sin30, 0.0)
grdb = Vector(cos30, -sin30, 0.0)
grdc = Vector(0.0, 1.0, 0.0)

cond = -1.0

#%%
# Mesh Points XY
zorg = 0.0
pnts = point_mesh_xy(0.0, 0.0, zorg, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy time elapsed is {elapsedc:.6f} seconds.')

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
phic = cupy_ctdp(pnts, grda, grdb, grdc, cond = cond)
velc = cupy_ctdv(pnts, grda, grdb, grdc)

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

phin = numpy_ctdp(pnts, grda, grdb, grdc, cond = cond)
veln = numpy_ctdv(pnts, grda, grdb, grdc)

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
# Mesh Points XY
zorg = -0.05
pnts = point_mesh_xy(0.0, 0.0, zorg, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy time elapsed is {elapsedc:.6f} seconds.')

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

# Doublet Velocity Velocity in X
axd, cfd = point_contourf_xy(pnts, flwn.vel.x)
_ = axd.set_title('3D Doublet Panel Velocity in X - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.x)
_ = axp.set_title('3D Doublet Panel Velocity in X - Cupy')

# Doublet Velocity Velocity in Y
axd, cfd = point_contourf_xy(pnts, flwn.vel.y)
_ = axd.set_title('3D Doublet Panel Velocity in Y - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.y)
_ = axp.set_title('3D Doublet Panel Velocity in Y - Cupy')

# Doublet Velocity Velocity in Z
axd, cfd = point_contourf_xy(pnts, flwn.vel.z)
_ = axd.set_title('3D Doublet Panel Velocity in Z - Numpy')

axp, cfp = point_contourf_xy(pnts, flwc.vel.z)
_ = axp.set_title('3D Doublet Panel Velocity in Z - Cupy')

#%%
# Check Components
phic = cupy_ctdp(pnts, grda, grdb, grdc, cond = cond)
velc = cupy_ctdv(pnts, grda, grdb, grdc)

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

phin = numpy_ctdp(pnts, grda, grdb, grdc, cond = cond)
veln = numpy_ctdv(pnts, grda, grdb, grdc)

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
# Mesh Points XZ
yorg = -0.05
pnts = point_mesh_zx(0.0, yorg, 0.0, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy time elapsed is {elapsedc:.6f} seconds.')

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
phic = cupy_ctdp(pnts, grda, grdb, grdc, cond = cond)
velc = cupy_ctdv(pnts, grda, grdb, grdc)

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

phin = numpy_ctdp(pnts, grda, grdb, grdc, cond = cond)
veln = numpy_ctdv(pnts, grda, grdb, grdc)

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
# Mesh Points YZ
xorg = -0.05
pnts = point_mesh_yz(xorg, 0.0, 0.0, num, num, 2.0, 2.0)

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedn = finish - start
print(f'Numpy time elapsed is {elapsedn:.6f} seconds.')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsedc = finish - start
print(f'Cupy time elapsed is {elapsedc:.6f} seconds.')

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
phic = cupy_ctdp(pnts, grda, grdb, grdc, cond = cond)
velc = cupy_ctdv(pnts, grda, grdb, grdc)

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

phin = numpy_ctdp(pnts, grda, grdb, grdc, cond = cond)
veln = numpy_ctdv(pnts, grda, grdb, grdc)

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
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Corner Point B
pnts = grdb

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Corner Point C
pnts = grdc

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Centre Point
pnts = (grda + grdb + grdc)/3.0

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Edge Point
pnts = (grda + grdb)/2.0

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')

#%%
# Outside Point
pnts = -grdc

start = perf_counter()
flwn = numpy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Numpy time elapsed is {elapsed:.6f} seconds.')

print(f'flwn = \n{flwn:.6f}\n')

start = perf_counter()
flwc = cupy_ctdf(pnts, grda, grdb, grdc, tol = 1e-12, cond = cond)
finish = perf_counter()
elapsed = finish - start
print(f'Cupy time elapsed is {elapsed:.6f} seconds.')

print(f'flwc = \n{flwc:.6f}\n')
