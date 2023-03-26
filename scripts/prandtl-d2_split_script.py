#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm import panelsystem_from_json
from pyapm.classes.horseshoevortex2d import HorseshoeVortex2D, Vector2D
from pyapm.outputs.msh import panelresult_to_msh
from matplotlib.pyplot import figure
from pygeom.matrix2d import zero_matrix_vector, elementwise_dot_product
from numpy.matlib import zeros

#%%
# Create Panel Mesh
jsonfilepath = '../files/Prandtl-D2_split.json'
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
# Profile Plots
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
for sct in psys.srfcs[0].scts[2:]:
    _ = ax.plot(sct.airfoil.x, sct.airfoil.y, label=sct.airfoil.name)
# _ = ax.legend()
