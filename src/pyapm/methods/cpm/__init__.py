from .constantgrid import ConstantGrid as ConstantGrid
from .constantpanel import ConstantPanel as ConstantPanel
from .constantsystem import ConstantSystem as ConstantSystem
from .constantwakepanel import ConstantWakePanel as ConstantWakePanel
from .constantresult import ConstantResult as ConstantResult
from .constantplot import ConstantPlot as ConstantPlot

cfrm = '.5f'
dfrm = '.7f'
efrm = '.4f'
sfrm = '.6f'

units = {
    'length': None,
    'mass': None,
    'force': None,
    'time': 's',
}

def set_unit_system(system: str):
    """Set the units for the output."""
    global units
    if system == 'metric' or system == 'SI':
        units['length'] = 'm'
        units['mass'] = 'kg'
        units['force'] = 'N'
    else:
        raise ValueError(f'Unknown unit system: {system}')

def get_unit_string(label: str) -> str:
    """Get the unit for a given label."""
    if label == 'length':
        lunit = units.get('length')
        if lunit is None:
            ustr = ''
        else:
            ustr = f' ({lunit})'
    elif label == 'mass':
        munit = units.get('mass')
        if munit is None:
            ustr = ''
        else:
            ustr = f' ({munit})'
    elif label == 'force':
        funit = units.get('force')
        if funit is None:
            ustr = ''
        else:
            ustr = f' ({funit})'
    elif label == 'moment':
        funit = units.get('force')
        lunit = units.get('length')
        if lunit is None or funit is None:
            ustr = ''
        else:
            ustr = f' ({funit}.{lunit})'
    elif label == 'area':
        lunit = units.get('length')
        if lunit is None:
            ustr = ''
        else:
            ustr = f' ({lunit}<sup>2</sup>)'
    elif label == 'velocity':
        lunit = units.get('length')
        tunit = units.get('time')
        if lunit is None or tunit is None:
            ustr = ''
        else:
            ustr = f' ({lunit}/{tunit})'
    elif label == 'density':
        munit = units.get('mass')
        lunit = units.get('length')
        if munit is None or lunit is None:
            ustr = ''
        else:
            ustr = f' ({munit}/{lunit}<sup>3</sup>)'
    elif label == 'pressure':
        funit = units.get('force')
        lunit = units.get('length')
        if funit is None or lunit is None:
            ustr = ''
        else:
            ustr = f' ({funit}/{lunit}<sup>2</sup>)'
    else:
        raise ValueError(f'Unknown label: {label}')
    return ustr

USE_CUPY = False

def set_cupy(use_cupy: bool = True):
    """Set whether to use cupy for calculations."""
    global USE_CUPY
    USE_CUPY = use_cupy
    if USE_CUPY:
        print('Using cupy for constant system calculations.')
    else:
        print('Using numpy for constant system calculations.')
