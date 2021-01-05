#%% Import Dependencies
from pyapm.tools.airfoil import airfoil_from_dat
from pyapm.tools.naca4 import NACA4
from matplotlib.pyplot import figure

#%% Create Airfoil
datfilepath = '../files/prandtl_root.dat'
airfoil = airfoil_from_dat(datfilepath)
airfoil.update(16)

naca4 = NACA4('0012', cnum=16)

print(f'airfoil xu = {airfoil.xu:}')
print(f'naca4 cdst = {naca4.cdst:}')
print(f'naca4 xc = {naca4.xc:}')
print(f'naca4 yc = {naca4.yc:}')
print(f'naca4 dydx = {naca4.dydx:}')
print(f'naca4 thc = {naca4.thc:}')
print(f'naca4 t = {naca4.t:}')
print(f'naca4 dtdx = {naca4.dtdx:}')
print(f'naca4 tht = {naca4.tht:}')

#%% Plot NACA
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
_ = ax.plot(naca4.x, naca4.y, '-o')

#%% Print Coordinates
for xi, yi in zip(naca4.xu, naca4.yu):
    print(f'{xi:.6f}, {yi:.6f}')

print(naca4.cdst)
