#%%
# Import Dependencies
from matplotlib.pyplot import figure
from pyapm.tools.naca4 import NACA4
from pyapm.tools.polyfoil import polyfoil_from_xy

#%%
# Create Airfoil
code = '2412'
name = 'NACA ' + code
num = 15
naca4 = NACA4(code, cnum = num)
x, z = naca4.x, naca4.y
pf = polyfoil_from_xy(name, x, z)
print(f'pf.a = {pf.a}')
print(f'pf.b0 = {pf.b0}')
print(f'pf.b = {pf.b}')

#%%
# Plot Airfoil
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
p = ax.plot(x, z, 'o', label='Airfoil')
pc = ax.plot(pf.xc, pf.yc, label='Fitted Camber')
pu = ax.plot(pf.xu, pf.yu, label='Fitted Upper')
pl = ax.plot(pf.xl, pf.yl, label='Fitted Lower')

#%%
# Create Airfoil
code = '0012'
name = 'NACA ' + code
num = 15
naca4 = NACA4(code, cnum = num)
x, z = naca4.x, naca4.y
pf = polyfoil_from_xy(name, x, z, na=0)
print(f'pf.a = {pf.a}')
print(f'pf.b0 = {pf.b0}')
print(f'pf.b = {pf.b}')

#%%
# Plot Airfoil
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
p = ax.plot(x, z, 'o', label='Airfoil')
pc = ax.plot(pf.xc, pf.yc, label='Fitted Camber')
pu = ax.plot(pf.xu, pf.yu, label='Fitted Upper')
pl = ax.plot(pf.xl, pf.yl, label='Fitted Lower')
