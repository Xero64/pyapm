#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm import panelsystem_from_json
from pyapm.classes.horseshoevortex2d import HorseshoeVortex2D, Vector2D
from pyapm.outputs.msh import panelresult_to_msh
from matplotlib.pyplot import figure
from pygeom.array2d import zero_arrayvector2d
from numpy import zeros

#%%
# Create Panel Mesh
jsonfilepath = '../files/Prandtl-D2.json'
psys = panelsystem_from_json(jsonfilepath)
display_markdown(psys)

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
psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%%
# Solve Panel Result
pres = psys.results['Design Point']

display_markdown(pres)
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

awh = zeros((len(hind), len(hind)), dtype=float)
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

hsvpnts = zero_arrayvector2d(num, dtype=float)
hsvnrms = zero_arrayvector2d(num, dtype=float)
for i, hsv in enumerate(hsv2ds):
    hsvpnts[i] = hsv.pnt
    hsvnrms[i] = hsv.nrm

awh2d = zeros((num, num), dtype=float)

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
