#%%
# Import Dependencies
from time import perf_counter
from IPython.display import display_markdown
from matplotlib.pyplot import figure
from numpy import zeros
from pyapm import PanelSystem
from pyapm.classes.horseshoevortex2d import HorseshoeVortex2D, Vector2D
from pyapm.outputs.k3d import PanelPlot
from pyapm.outputs.msh import panelresult_to_msh
from pygeom.geom2d import Vector2D
from pyapm import set_cupy

set_cupy(True)  # Set to True if you want to use CuPy for GPU acceleration

#%%
# Create Panel Mesh
start = perf_counter()
jsonfilepath = '../files/Prandtl-D2.json'
psys = PanelSystem.from_json(jsonfilepath)
finish = perf_counter()
elapsed = finish - start
print(f'Panel System created in {elapsed:.2f} seconds')
display_markdown(psys)

psys.save_initial_state(jsonfilepath)

#%%
# System Plots
axt1 = psys.plot_twist_distribution()
_ = axt1.set_ylabel('Twist (deg)')
_ = axt1.set_xlabel('Span-Wise Coordinate - b (m)')
axt2 = psys.plot_tilt_distribution()
_ = axt2.set_ylabel('Tilt (deg)')
_ = axt2.set_xlabel('Span-Wise Coordinate - b (m)')
axt = psys.plot_chord_distribution()
_ = axt.set_ylabel('Chord (m)')
_ = axt.set_xlabel('Span-Wise Coordinate - b (m)')
axw = psys.plot_strip_width_distribution()
_ = axw.set_ylabel('Strip Width (m)')
_ = axw.set_xlabel('Span-Wise Coordinate - b (m)')

#%%
# Assembly and Solution
psys.assemble_horseshoes_wash()
# psys.assemble_panels_wash()
psys.assemble_panels_phi()
psys.assemble_horseshoes_phi()
psys.solve_system()

#%%
# Solve Panel Result
pres = psys.results['Design Point']

display_markdown(pres)
display_markdown(pres.stability_derivatives)
display_markdown(pres.control_derivatives)
display_markdown(pres.surface_loads)

mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%%
# Distribution Plots
axd = pres.plot_strip_drag_force_distribution()
_ = axd.set_ylabel('Drag Force (N/m)')
_ = axd.set_xlabel('Span-Wise Coordinate - b (m)')
_ = pres.plot_trefftz_drag_force_distribution(ax=axd)
axs = pres.plot_strip_side_force_distribution()
_ = axs.set_ylabel('Side Force (N/m)')
_ = axs.set_xlabel('Span-Wise Coordinate - b (m)')
_ = pres.plot_trefftz_side_force_distribution(ax=axs)
axl = pres.plot_strip_lift_force_distribution()
_ = axl.set_ylabel('Lift Force (N/m)')
_ = axl.set_xlabel('Span-Wise Coordinate - b (m)')
_ = pres.plot_trefftz_lift_force_distribution(ax=axl)
axc = pres.plot_trefftz_circulation_distribution()
_ = axc.set_ylabel('Circulation (m^2/s)')
_ = axc.set_xlabel('Span-Wise Coordinate - b (m)')
axw = pres.plot_trefftz_wash_distribution()
_ = axw.set_ylabel('Wash (m/s)')
_ = axw.set_xlabel('Span-Wise Coordinate - b (m)')

#%%
# Extra Plots
fig = figure(figsize=(12, 8))
ax = fig.gca()

# hsvs = psys.hsvs[1::2]
hsv2ds = []

ylst = []
zlst = []

ypos = []
hind = []
for i, strp in enumerate(psys.strps):
    pnla = strp.pnls[0]
    pinda = pnla.ind
    hindsa = psys.phind[pinda]
    hind.append(hindsa[0])
    hsv = psys.hsvs[hindsa[0]]
# for hsv in hsvs:
    y = [hsv.grda.y, hsv.grdb.y]
    z = [hsv.grda.z, hsv.grdb.z]
    ax.plot(y, z)
    ypos.append(hsv.pnto.y)
    ylst += y
    zlst += z

ylst = ylst[0::2] + [ylst[-1]]
# print(ylst)

zlst = zlst[0::2] + [zlst[-1]]
# print(zlst)

for i in range(len(ylst)-1):
    grda = Vector2D(ylst[i], zlst[i])
    grdb = Vector2D(ylst[i+1], zlst[i+1])
    hsv2ds.append(HorseshoeVortex2D(grda, grdb))

circ = pres.ffres.circ.transpose().tolist()[0]

lift = [pres.rho*pres.speed*circi for circi in circ]

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(ypos, circ)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(ypos, lift)

awh = zeros((len(hind), len(hind)))
for i, hindi in enumerate(hind):
    for j, hindj in enumerate(hind):
        awh[i, j] = psys.awh[hindi, hindj]

wash = awh*pres.ffres.circ

wash = wash.transpose().tolist()[0]

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.plot(ypos, wash)

#%%
# Horseshoe Vortex 2D
num = len(hsv2ds)

hsvpnts = Vector2D.zeros(num)
hsvnrms = Vector2D.zeros(num)
for i, hsv in enumerate(hsv2ds):
    hsvpnts[i] = hsv.pnt
    hsvnrms[i] = hsv.nrm

awh2d = zeros((num, num))

for i, hsv in enumerate(hsv2ds):
    avh = hsv.induced_velocity(hsvpnts)
    awh2d[:, i] = hsvnrms.dot(avh)
    # awh2d[:, i] = avh.y

wash2d = awh2d@pres.ffres.circ

wash2d = wash2d.transpose().tolist()[0]

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.plot(ypos, wash2d)

#%%
# Display Result
pnlpl = PanelPlot(psys, pres)

mshplot = pnlpl.plot()
mshplot += pnlpl.panel_mesh()
mshplot.display()

sigplot = pnlpl.plot()
sigplot += pnlpl.panel_sigma_plot()
sigplot.display()

siggplot = pnlpl.plot()
siggplot += pnlpl.grid_sigma_plot()
siggplot.display()

muplot = pnlpl.plot()
muplot += pnlpl.panel_mu_plot()
muplot.display()

mugplot = pnlpl.plot()
mugplot += pnlpl.grid_mu_plot()
mugplot.display()

#%%
# Longitudinal Stability Analysis
pres = psys.results['Zero Point']
wing_force = pres.nfres.nffrctot
wing_moment = pres.nfres.nfmomtot
wing_force_alpha = pres.stres.alpha.dfrctot
wing_moment_alpha = pres.stres.alpha.dmomtot
el = 0.0

print(f'Wing Force = {wing_force:.4f} N\n')
print(f'Wing Moment = {wing_moment:.4f} N.m\n')
print(f'Wing Force Alpha = {wing_force_alpha:.4f} N/rad\n')
print(f'Wing Moment Alpha = {wing_moment_alpha:.4f} N.m/rad\n')

Cz_wing = wing_force.z/pres.qfs/psys.sref
Cm_wing = wing_moment.y/pres.qfs/psys.sref/psys.cref
Cza_wing = wing_force_alpha.z/pres.qfs/psys.sref
Cma_wing = wing_moment_alpha.y/pres.qfs/psys.sref/psys.cref
al0_wing = -Cz_wing/Cza_wing

print(f'Cz Wing = {Cz_wing:.6f}\n')
print(f'Cm Wing = {Cm_wing:.6f}\n')
print(f'Cza Wing = {Cza_wing:.6f}\n')
print(f'Cma Wing = {Cma_wing:.6f}\n')

xcg = psys.rref.x
xac = xcg - Cma_wing/Cza_wing*psys.cref
xac_wing = xcg - Cma_wing/Cza_wing*psys.cref

xm0 = xcg - Cm_wing/Cz_wing*psys.cref
xm0_wing = xcg - Cm_wing/Cz_wing*psys.cref

Cm0_wing = Cm_wing + Cz_wing*(xac_wing - xm0_wing)/psys.cref

print(f'xcg = {xcg:.6f} m\n')
print(f'xac = {xac:.6f} m, xm0 = {xm0:.6f} m\n')
print(f'xac Wing = {xac_wing:.6f} m, xm0 Wing = {xm0_wing:.6f} m, Cm0 Wing = {Cm0_wing:.6f}\n')
