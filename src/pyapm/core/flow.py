from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from numpy import (bool_, copy, logical_and, logical_or, ndim, ravel, repeat,
                   reshape, result_type, shape, size, split, stack, sum,
                   transpose, zeros)
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class Flow():
    """Flow Class"""
    phi: 'NDArray' = None
    vel: Vector = None

    def __init__(self, phi: 'NDArray', vel: Vector) -> None:
        self.phi = phi
        self.vel = vel

    def to_tuple(self) -> tuple['NDArray', Vector]:
        """Returns the phi and vel values of this flow."""
        return self.phi, self.vel

    def __mul__(self, obj: Any) -> 'Flow':
        phi = self.phi*obj
        vel = self.vel*obj
        return Flow(phi, vel)

    def __rmul__(self, obj: Any) -> 'Flow':
        phi = obj*self.phi
        vel = obj*self.vel
        return Flow(phi, vel)

    def __truediv__(self, obj: Any) -> 'Flow':
        phi = self.phi/obj
        vel = self.vel/obj
        return Flow(phi, vel)

    def __pow__(self, obj: Any) -> 'Flow':
        phi = self.phi**obj
        vel = self.vel**obj
        return Flow(phi, vel)

    def __rpow__(self, obj: Any) -> 'Flow':
        phi = obj**self.phi
        vel = obj**self.vel
        return Flow(phi, vel)

    def __add__(self, obj: 'Flow') -> 'Flow':
        try:
            phi = self.phi + obj.phi
            vel = self.vel + obj.vel
            return Flow(phi, vel)
        except AttributeError:
            err = 'Flow object can only be added to Flow object.'
            raise TypeError(err)

    def __sub__(self, obj: 'Flow') -> 'Flow':
        try:
            phi = self.phi - obj.phi
            vel = self.vel - obj.vel
            return Flow(phi, vel)
        except AttributeError:
            err = 'Flow object can only be subtracted from Flow object.'
            raise TypeError(err)

    def __pos__(self) -> 'Flow':
        return self

    def __neg__(self) -> 'Flow':
        return Flow(-self.phi, -self.vel)

    def __repr__(self) -> str:
        if self.ndim == 0:
            return f'<Flow: {self.phi:}, {self.vel:}>'
        else:
            return f'<Flow shape: {self.shape:}, dtype: {self.dtype}>'

    def __str__(self) -> str:
        if self.ndim == 0:
            outstr = f'Flow: {self.phi:}, {self.vel:}'
        else:
            outstr = f'Flow shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'phi:\n{self.phi:}\nvel:\n{self.vel:}\n'
        return outstr

    def __format__(self, frm: str) -> str:
        if self.ndim == 0:
            outstr = f'Flow: {self.phi:{frm}}, {self.vel:{frm}}'
        else:
            outstr = f'Flow shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'phi:\n{self.phi:{frm}}\nvel:\n{self.vel:{frm}}\n'
        return outstr

    def __matmul__(self, obj: 'NDArray') -> 'Flow':
        try:
            phi = self.phi@obj
            vel = self.vel@obj
            return Flow(phi, vel)
        except AttributeError:
            err = 'ArrayFlow2D object can only be multiplied by a numpy array.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'NDArray') -> 'Flow':
        try:
            phi = obj@self.phi
            vel = obj@self.vel
            return Flow(phi, vel)
        except AttributeError:
            err = 'ArrayFlow2D object can only be multiplied by a numpy array.'
            raise TypeError(err)

    def rmatmul(self, mat: 'NDArray') -> 'Flow':
        """Returns the right matrix multiplication of this flow."""
        return self.__rmatmul__(mat)

    def __getitem__(self, key) -> 'Flow':
        phi = self.phi[key]
        vel = self.vel[key]
        return Flow(phi, vel)

    def __setitem__(self, key, value: 'Flow') -> None:
        try:
            self.phi[key] = value.phi
            self.vel[key] = value.vel
        except AttributeError:
            err = 'Flow index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> tuple[int, ...]:
        shape_phi = shape(self.phi)
        shape_vel = self.vel.shape
        if shape_phi == shape_vel:
            return shape_phi
        else:
            raise ValueError('Flow phi and vel should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        return result_type(self.phi, self.vel.x, self.vel.y, self.vel.z)

    @property
    def ndim(self) -> int:
        ndim_phi = ndim(self.phi)
        ndim_vel = self.vel.ndim
        if ndim_phi == ndim_vel:
            return ndim_phi
        else:
            raise ValueError('Flow phi and vel should have the same ndim.')

    @property
    def size(self) -> int:
        size_phi = size(self.phi)
        size_vel = self.vel.size
        if size_phi == size_vel:
            return size_phi
        else:
            raise ValueError('Flow phi and vel should have the same size.')

    def transpose(self, **kwargs: dict[str, Any]) -> 'Flow':
        phi = transpose(self.phi, **kwargs)
        vel = self.vel.transpose(**kwargs)
        return Flow(phi, vel)

    def sum(self, **kwargs: dict[str, Any]) -> 'Flow':
        phi = sum(self.phi, **kwargs)
        vel = self.vel.sum(**kwargs)
        return Flow(phi, vel)

    def repeat(self, repeats, axis=None) -> 'Flow':
        phi = repeat(self.phi, repeats, axis=axis)
        vel = self.vel.repeat(repeats, axis=axis)
        return Flow(phi, vel)

    def reshape(self, shape, order='C') -> 'Flow':
        phi = reshape(self.phi, shape, order=order)
        vel = self.vel.reshape(shape, order=order)
        return Flow(phi, vel)

    def ravel(self, order='C') -> 'Flow':
        phi = ravel(self.phi, order=order)
        vel = self.vel.ravel(order=order)
        return Flow(phi, vel)

    def copy(self, order='C') -> 'Flow':
        phi = copy(self.phi, order=order)
        vel = self.vel.copy(order=order)
        return Flow(phi, vel)

    def split(self, numsect: int,
              axis: int = -1) -> Iterable['Flow']:
        philst = split(self.phi, numsect, axis=axis)
        vellst = self.vel.split(numsect, axis=axis)
        for phii, veli in zip(philst, vellst):
            yield Flow(phii, veli)

    def unpack(self) -> Iterable['Flow']:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        philst = split(self.phi, numsect, axis=-1)
        vellst = self.vel.split(numsect, axis=-1)
        for phii, veli in zip(philst, vellst):
            yield Flow(phii, veli).reshape(shape)

    def __next__(self) -> 'Flow':
        return Flow(next(self.phi), next(self.vel))

    def __eq__(self, obj: 'Flow') -> 'NDArray[bool_]':
        try:
            phieq = self.phi == obj.phi
            veleq = self.vel == obj.vel
            return logical_and(phieq, veleq)
        except AttributeError:
            return False

    def __neq__(self, obj: 'Flow') -> 'NDArray[bool_]':
        try:
            phineq = self.phi != obj.phi
            velneq = self.vel != obj.vel
            return logical_or(phineq, velneq)
        except AttributeError:
            return False

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = (),
              **kwargs: dict[str, Any]) -> 'Flow':
        phi = zeros(shape, **kwargs)
        vel = Vector.zeros(shape, **kwargs)
        return cls(phi, vel)

    @classmethod
    def stack(cls, flows: Iterable['Flow'],
              axis: int = -1) -> 'Flow':
        phi = stack([flow.phi for flow in flows], axis=axis)
        vel = Vector.stack([flow.vel for flow in flows], axis=axis)
        return cls(phi, vel)
