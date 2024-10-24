#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray
from pygeom.geom2d import CubicSpline2D, Vector2D

from pyapm.tools.naca4 import NACA4

#%%
# Create NACA4
naca4 = NACA4('5414', cnum=16)
pnts = Vector2D(asarray(naca4.x), asarray(naca4.y))
spline = CubicSpline2D(pnts)

#%%
# Normal Vectors
nrms = spline.d2r.to_unit()
u = [n.x for n in nrms]
v = [n.y for n in nrms]

num = len(naca4.xl)

nrml = nrms[num::-1]
nrmu = nrms[-num:]

splines: list[CubicSpline2D] = []
for i in range(1, len(naca4.xl)-1):
    pntul = Vector2D.zeros(2)
    pntul[0] = Vector2D(naca4.xl[i], naca4.yl[i])
    pntul[1] = Vector2D(naca4.xu[i], naca4.yu[i])
    splines.append(CubicSpline2D(pntul, bctype = ((1, nrml[i]), (1, -nrmu[i]))))

#%%
# Plot Spline
pntsp = spline.evaluate_points(num=10)
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
_ = ax.plot(pntsp.x, pntsp.y, color='blue')
for spl in splines:
    pntsp = spl.evaluate_points(num=10)
    _ = ax.plot(pntsp.x, pntsp.y, color='green')
