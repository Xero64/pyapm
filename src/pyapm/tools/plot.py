from typing import TYPE_CHECKING

from matplotlib.pyplot import figure
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    from numpy.typing import NDArray


def point_contourf_xy(pnts: Vector, res: 'NDArray',
                      figsize: tuple[int]=(10, 8), **kwargs) -> tuple['Axes',
                                                                      'QuadContourSet']:

    kwargs.setdefault('levels', 20)
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('extend', 'both')

    fig = figure(figsize=figsize)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cf = ax.contourf(pnts.x, pnts.y, res, **kwargs)
    fig.colorbar(cf)
    return ax, cf

def point_contourf_yx(pnts: Vector, res: 'NDArray',
                      figsize: tuple[int]=(10, 8), **kwargs) -> tuple['Axes',
                                                                      'QuadContourSet']:

    kwargs.setdefault('levels', 20)
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('extend', 'both')

    fig = figure(figsize=figsize)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_yaxis()
    cf = ax.contourf(pnts.y, pnts.x, res, **kwargs)
    fig.colorbar(cf, location='bottom')
    return ax, cf

def point_contourf_yz(pnts: Vector, res: 'NDArray',
                      figsize: tuple[int]=(10, 8), **kwargs) -> tuple['Axes',
                                                                      'QuadContourSet']:

    kwargs.setdefault('levels', 20)
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('extend', 'both')

    fig = figure(figsize=figsize)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    cf = ax.contourf(pnts.y, pnts.z, res, **kwargs)
    fig.colorbar(cf)
    return ax, cf

def point_contourf_zx(pnts: Vector, res: 'NDArray',
                      figsize: tuple[int]=(10, 8), **kwargs) -> tuple['Axes',
                                                                      'QuadContourSet']:

    kwargs.setdefault('levels', 20)
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('extend', 'both')

    fig = figure(figsize=figsize)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    cf = ax.contourf(pnts.z, pnts.x, res, **kwargs)
    fig.colorbar(cf)
    return ax, cf

def point_contourf_xz(pnts: Vector, res: 'NDArray',
                      figsize: tuple[int]=(10, 8), **kwargs) -> tuple['Axes',
                                                                      'QuadContourSet']:

    kwargs.setdefault('levels', 20)
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('extend', 'both')

    fig = figure(figsize=figsize)
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    cf = ax.contourf(pnts.x, pnts.z, res, **kwargs)
    fig.colorbar(cf)
    return ax, cf

def plot_line(x: 'NDArray', y: 'NDArray', ax=None,
              figsize: tuple[int]=(10, 8), **kwargs):

    kwargs['extend'] = kwargs.get('extend', 'both')

    x = x.tolist()[0]
    y = y.tolist()[0]
    if ax is None:
        fig = figure(figsize=figsize)
        ax = fig.gca()
        ax.grid(True)
    cf = ax.plot(x, y, **kwargs)
    return ax, cf
