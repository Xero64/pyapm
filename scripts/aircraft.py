#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh

#%%
# Import Geometry
jsonfilepath = '../files/Aircraft.json'
psys = panelsystem_from_json(jsonfilepath)

#%%
# Display System
display_markdown(psys)

#%%
# Display Results
for case in psys.results:
    pres = psys.results[case]
    display_markdown(pres)

#%%
# Mesh File Output
ptrm = psys.results['Positive 1g Cruise']
panelresult_to_msh(ptrm, '../results/Aircraft.msh')
display_markdown(ptrm)
display_markdown(ptrm.surface_loads)
display_markdown(ptrm.stability_derivatives)
display_markdown(ptrm.control_derivatives)

# #%%
# # Deflect Elevator
# pres = ptrm.to_result('Positive 1g Cruise + Controls 25 deg')
# pres.set_controls(elevator=25.0, aileron=25.0, rudder=25.0)
# panelresult_to_msh(pres, '../results/Aircraft_Trim.msh')
# display_markdown(pres)
# display_markdown(pres.surface_loads)
# display_markdown(pres.stability_derivatives)
# display_markdown(pres.control_derivatives)

# #%%
# # Plot Lift Distribution
# axl = ptrm.plot_trefftz_lift_force_distribution()
# axl = ptrm.plot_strip_lift_force_distribution(ax=axl)
# axl = pres.plot_trefftz_lift_force_distribution(ax=axl)
# axl = pres.plot_strip_lift_force_distribution(ax=axl)

# #%%
# # Plot Y Force Distribution
# axy = ptrm.plot_trefftz_side_force_distribution()
# axy = ptrm.plot_strip_side_force_distribution(ax=axy)
# axy = pres.plot_trefftz_side_force_distribution(ax=axy)
# axy = pres.plot_strip_side_force_distribution(ax=axy)

# #%%
# # Plot Drag Distribution
# axd = ptrm.plot_trefftz_drag_force_distribution()
# axd = ptrm.plot_strip_drag_force_distribution(ax=axd)
# axd = pres.plot_trefftz_drag_force_distribution(ax=axd)
# axd = pres.plot_strip_drag_force_distribution(ax=axd)

# #%%
# # Display Derivatives
# display_markdown(ptrm.stability_derivatives)
# display_markdown(ptrm.control_derivatives)

# #%%
# # 60deg Banked Turn
# ptrm = psys.results['60deg Banked Turn Dive']
# panelresult_to_msh(ptrm, '../results/Aircraft_Bank.msh')
# display_markdown(ptrm)
# display_markdown(pres.surface_loads)
# display_markdown(pres.stability_derivatives)
# display_markdown(pres.control_derivatives)

# #%%
# # Plot Lift Distribution
# axl = ptrm.plot_trefftz_lift_force_distribution()
# axl = ptrm.plot_strip_lift_force_distribution(ax=axl)

# #%%
# # Plot Y Force Distribution
# axy = ptrm.plot_trefftz_side_force_distribution()
# axy = ptrm.plot_strip_side_force_distribution(ax=axy)

# #%%
# # Plot Drag Distribution
# axd = ptrm.plot_trefftz_drag_force_distribution()
# axd = ptrm.plot_strip_drag_force_distribution(ax=axd)
