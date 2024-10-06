#%%
# Import Dependencies
from matplotlib.pyplot import figure

from pyapm.tools import read_dat
from pyapm.tools.polyfoil import polyfoil_from_xy

#%%
# Number of polynomial terms
np = 10

#%%
# Create Airfoil
name, x, y = read_dat('../files/root_airfoil.dat')
pf = polyfoil_from_xy(name, x, y, na=np, nb=np)
print(f'pf.a = {pf.a}')
print(f'pf.b0 = {pf.b0}')
print(f'pf.b = {pf.b}')

#%%
# Plot Airfoil
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
p = ax.plot(x, y, 'o', label=name)
pc = ax.plot(pf.xc, pf.yc, label='Fitted Camber')
pu = ax.plot(pf.xu, pf.yu, label='Fitted Upper')
pl = ax.plot(pf.xl, pf.yl, label='Fitted Lower')

#%%
# Create Airfoil
name, x, y = read_dat('../files/tip_airfoil.dat')
pf = polyfoil_from_xy(name, x, y, na=0, nb=np)
print(f'pf.a = {pf.a}')
print(f'pf.b0 = {pf.b0}')
print(f'pf.b = {pf.b}')

#%%
# Plot Airfoil
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
p = ax.plot(x, y, 'o', label=name)
pc = ax.plot(pf.xc, pf.yc, label='Fitted Camber')
pu = ax.plot(pf.xu, pf.yu, label='Fitted Upper')
pl = ax.plot(pf.xl, pf.yl, label='Fitted Lower')
