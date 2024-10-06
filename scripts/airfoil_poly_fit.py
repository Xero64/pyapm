#%%
# Import Dependencies
from matplotlib.pyplot import figure
from pyfoil.airfoil.naca import naca_to_xyt

from pyapm.tools.polyfoil import polyfoil_from_xy

#%%
# Create Airfoil
code = '2412'
name = 'NACA ' + code
num = 15
x, y, t = naca_to_xyt(code, num)
pf = polyfoil_from_xy(name, x, y)
print(f'pf.a = {pf.a}')
print(f'pf.b0 = {pf.b0}')
print(f'pf.b = {pf.b}')

#%%
# Plot Airfoil
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
p = ax.plot(x, y, 'o', label='Airfoil')
pc = ax.plot(pf.xc, pf.yc, label='Fitted Camber')
pu = ax.plot(pf.xu, pf.yu, label='Fitted Upper')
pl = ax.plot(pf.xl, pf.yl, label='Fitted Lower')

#%%
# Create Airfoil
code = '0012'
name = 'NACA ' + code
num = 15
x, y, t = naca_to_xyt(code, num)
pf = polyfoil_from_xy(name, x, y, na=0)
print(f'pf.a = {pf.a}')
print(f'pf.b0 = {pf.b0}')
print(f'pf.b = {pf.b}')

#%%
# Plot Airfoil
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
p = ax.plot(x, y, 'o', label='Airfoil')
pc = ax.plot(pf.xc, pf.yc, label='Fitted Camber')
pu = ax.plot(pf.xu, pf.yu, label='Fitted Upper')
pl = ax.plot(pf.xl, pf.yl, label='Fitted Lower')
