#%%
# Import Dependencies
from pygeom.geom2d import CubicSpline2D, Point2D, Vector2D
from pyapm.tools.naca4 import NACA4
from matplotlib.pyplot import figure

#%%
# Create NACA4
naca4 = NACA4('5414', cnum=16)
pnts = [Point2D(x, y) for x, y in zip(naca4.x, naca4.y)]
spline = CubicSpline2D(pnts)

#%%
# Normal Vectors
nrms = [Vector2D(-dr.y, dr.x).to_unit() for dr in spline.dr]
u = [n.x for n in nrms]
v = [n.y for n in nrms]

num = len(naca4.xl)

nrml = nrms[0:num]
nrml.reverse()
nrmu = nrms[-num:]

splines = []
for i in range(1, len(naca4.xl)-1):
    pntl = Point2D(naca4.xl[i], naca4.yl[i])
    pntu = Point2D(naca4.xu[i], naca4.yu[i])
    splines.append(CubicSpline2D([pntl, pntu], tanA=-nrml[i], tanB=nrmu[i]))

#%%
# Plot Spline
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
# _ = ax.quiver(naca4.x, naca4.y, u, v)
_ = spline.plot_spline(num=10, ax=ax)
for spl in splines:
    _ = spl.plot_spline(num=10, ax=ax, color='green')
