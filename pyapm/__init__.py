from .classes.panelsystem import PanelSystem as PanelSystem

USE_CUPY = False

def set_cupy(use_cupy: bool = True):
    """Set whether to use cupy for calculations."""
    global USE_CUPY
    USE_CUPY = use_cupy
    if USE_CUPY:
        print('Using cupy for calculations.')
    else:
        print('Using numpy for calculations.')
