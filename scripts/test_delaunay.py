#%% Import Dependencies
# import scipy.spatial.Delaunay as Delaunay
from matplotlib.pyplot import figure
from matplotlib.tri import Triangulation
from pyapm.tools.naca4 import NACA4

#%% Create Triangulation
x = [0.0, 0.0, 1.0, 1.0]
y = [0.0, 1.1, 0.0, 1.0]
# points = [(xi, yi) for xi, yi in zip(x, y)]
tri = Triangulation(x, y)

#%% Plot Triangulation
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
_ = ax.triplot(tri)
_ = ax.plot(x, y, 'o')

#%% Create Triangulation
naca4 = NACA4('5414', cnum=20)
x = naca4.x + naca4.xc[4:-2]
y = naca4.y + naca4.yc[4:-2]
# points = [(xi, yi) for xi, yi in zip(x, y)]
# tri = Delaunay(points)
tri = Triangulation(x, y)

#%% Plot Triangulation
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
_ = ax.triplot(tri)
_ = ax.plot(x, y, 'o')
