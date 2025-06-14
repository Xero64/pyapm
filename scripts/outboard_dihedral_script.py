#%%
# Load Dependencies
from IPython.display import display_markdown
from pyapm import PanelSystem
from pyapm.outputs.msh import panelresult_to_msh

#%%
# Create Lattice System
jsonfilepath = '../files/Test_outboard_dihedral.json'
psys = PanelSystem.from_json(jsonfilepath)
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
# Solve Panel Result
pres = psys.results['Test Alpha']

display_markdown(pres)

mshfilepath = '../results/'+ psys.name + ' - ' + pres.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%%
# Distribution Plots
axd = pres.plot_strip_drag_force_distribution()
axd = pres.plot_trefftz_drag_force_distribution(ax=axd)
_ = axd.set_ylabel('Drag Force (N/m)')
_ = axd.set_xlabel('Span-Wise Coordinate - b (m)')
axs = pres.plot_strip_side_force_distribution()
axs = pres.plot_trefftz_side_force_distribution(ax=axs)
_ = axs.set_ylabel('Side Force (N/m)')
_ = axs.set_xlabel('Span-Wise Coordinate - b (m)')
axl = pres.plot_strip_lift_force_distribution()
axl = pres.plot_trefftz_lift_force_distribution(ax=axl)
_ = axl.set_ylabel('Lift Force (N/m)')
_ = axl.set_xlabel('Span-Wise Coordinate - b (m)')
axc = pres.plot_trefftz_circulation_distribution()
_ = axc.set_ylabel('Circulation (m^2/s)')
_ = axc.set_xlabel('Span-Wise Coordinate - b (m)')
axw = pres.plot_trefftz_wash_distribution()
_ = axw.set_ylabel('Wash (m/s)')
_ = axw.set_xlabel('Span-Wise Coordinate - b (m)')
