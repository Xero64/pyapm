# pyapm
An Aerodynamic Panel Method implemented in Python for use from Python scripts. Examples in the "./scripts" folder.

**JSON Definition File:**

```json
{
    "name": "Prandtl-D2",
    "mach": 0.0,
    "sref": 0.94064328,
    "cref": 0.2686,
    "bref": 3.749,
    "xref": 0.3270,
    "yref": 0.0,
    "zref": 0.0,
    "surfaces": [
        {
            "name": "Wing",
            "twist": 0.0,
            "mirror": true,
            "cnum": 16,
            "sections": [
                {
                    "xpos": 0.0,
                    "ypos": 0.0,
                    "zpos": 0.0,
                    "chord": 0.40005,
                    "bnum": 50,
                    "bspc": "full-cosine",
                    "airfoil": "prandtl_root.dat",
                    "xoc": 0.0,
                    "zoc": 0.0
                },
                {
                    "xpos": 0.83459,
                    "ypos": 1.87452,
                    "zpos": 0.08177,
                    "chord": 0.10008,
                    "airfoil": "prandtl_tip.dat",
                    "xoc": 0.0,
                    "zoc": 0.0
                }
            ],
            "functions": [
                {
                    "variable": "twist",
                    "spacing": "equal",
                    "interp": "cubic",
                    "values": [
                        8.3274,
                        8.5524,
                        8.7259,
                        8.8441,
                        8.9030,
                        8.8984,
                        8.8257,
                        8.6801,
                        8.4565,
                        8.1492,
                        7.7522,
                        7.2592,
                        6.6634,
                        5.9579,
                        5.1362,
                        4.1927,
                        3.1253,
                        1.9394,
                        0.6589,
                        -0.6417,
                        -1.6726
                    ]
                }
            ]
        }
    ],
    "cases": [
        {
            "name": "Design Point",
            "alpha": 0.0,
            "speed": 13.0,
            "density": 1.145
        }
    ]
}

```

**Typical Python Script File "test_prandtl-d2.py":**

```python
#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelResult, PanelSystem
from pyapm.output.msh import panelresult_to_msh

#%%
# Create Panel Mesh
jsonfilepath = r'../files/Prandtl-D2.json'
psys = PanelSystem.from_json(jsonfilepath)
psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%%
# Solve Panel Result
alpha = 0.0
speed = 13.0
rho = 1.145

pres = PanelResult('Design Point', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed)

display_markdown(pres)
display_markdown(pres.surface_loads)

mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

```

**Mesh File Output:**

You can generate a Gmsh mesh file (*.msh) directly from a Python script using the following code snippet.

```python
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
```

This will output a mesh file to the specified location, which can then be viewed in Gmsh. The latest version of Gmsh can be downloaded at:

http://gmsh.info/

Use File > Open in Gmsh to open the mesh file with the pressure results.

A sample of the aircraft shown in Gmsh is captured below. Consult Gmsh help to operate Gmsh.

![](https://github.com/Xero64/pyapm/raw/main/Readme.png)
