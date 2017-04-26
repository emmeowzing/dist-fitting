#! /usr/bin/env python3.6
# -*- coding: utf-8 -*-

from typing import Tuple
from types import FunctionType as Function
from functools import wraps
from scipy.integrate import dblquad
from math import pi

import numpy as np

_either = any


def scale(f: Function) -> Function:
    """ Decorator to conserve information """
    @wraps(f)
    def _new_f(*args, **kwargs) -> np.ndarray:
        arr = f(*args, **kwargs)
        # Scale array by volume (in this case \sum_{i,j}k_{i,j} in the discrete case
        # is the same as a double integral to find volume on
        # \iint_{[-3 * sigma, 3 * sigma]} in the continuous case/
        return arr / np.sum(arr)
    return _new_f


class Kernel:
    """ Generate different kernels for testing """
    def __init__(self, dims: Tuple[int, int]) -> None:

        # OaA convolution requires, in the way I've written it, a kernel with even
        # dimensions
        if _either(map(lambda x: x % 2, dims)):
            raise ValueError(
                'Kernel expected even dims, received %s' % (dims,)
            )

        self.dims = dims

    @scale
    def gaussian(self, sigma: float =1.0) -> np.ndarray:
        kernel = np.empty(self.dims)

        unit_square = (-0.5, 0.5, lambda y: -0.5, lambda y: 0.5)

        x_shift = 1.0 if self.dims[0] % 2 else 0.5
        y_shift = 1.0 if self.dims[1] % 2 else 0.5

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                # integrate on a unit square centered at the origin as the
                # function moves about it in discrete unit steps
                res = dblquad(
                    lambda x, y: 1 / (2 * pi * sigma ** 2) * np.exp(
                        - ((x + i - self.dims[0] // 2 + x_shift) / sigma) ** 2
                        - ((y + j - self.dims[1] // 2 + y_shift) / sigma) ** 2),
                    *unit_square
                )[0]

                kernel[i][j] = res

        return kernel

    def laplacian(self) -> np.ndarray:
        return np.array([[ 0, -1,  0],
                         [-1,  4, -1],
                         [ 0, -1,  0]]) / 4

    @scale
    def identity(self) -> np.ndarray:
        """ The identity kernel (there's really no need to scale this) """
        id = np.zeros(self.dims)
        id[self.dims[0] // 2, self.dims[1] // 2] = 1.0
        return id

    @scale
    def moffat(self, alpha: float =3.0, beta: float =2.5) -> np.ndarray:
        """ The Gaussian is a limiting case of this kernel as \beta -> \infty """
        kernel = np.zeros(self.dims)
        unit_square = (-0.5, 0.5, lambda y: -0.5, lambda y: 0.5)

        x_shift = 1.0 if self.dims[0] % 2 else 0.5
        y_shift = 1.0 if self.dims[1] % 2 else 0.5

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                res = dblquad(
                    lambda x, y: (beta - 1) / (pi * alpha ** 2) * \
                        (1 + ((x + i - self.dims[0] // 2 + x_shift) ** 2
                            + (y + j - self.dims[1] // 2 + y_shift) ** 2)
                         / (alpha ** 2)) ** (-beta),
                    *unit_square
                )[0]

                kernel[i][j] = res

        return kernel

    ## Add more kernels here