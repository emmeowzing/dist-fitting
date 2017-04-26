#! /usr/bin/env python3.6
# -*- coding: utf-8 -*-

""" Fit a Gaussian and a Moffat function to pixels """

from typing import List
from types import FunctionType as Function
from math import exp, pi, sqrt, log
from numpy.linalg import norm
from numba import jit
from random import random
from itertools import product
from warnings import warn
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from photutils.background import SExtractorBackground as SEbkd
from psf import Kernel

import numpy as np
import os


@jit
def gaussian(X: int, Y: int, alpha: float, mu_X: float, mu_Y: float, sigma: float) -> float:
    # compute the 2d gaussian function at this point
    res = alpha * exp(-(((X - mu_X) / sigma) ** 2 + ((Y - mu_Y) / sigma) ** 2) / 2) / (2 * pi * sigma * sigma)

    return res


@jit
def partialalphagaussian(X: int, Y: int, *args: float) -> float:
    return gaussian(X, Y, *args) / args[0]


@jit
def partialsigmagaussian(X: int, Y: int, *args: float) -> float:
    alpha, mu_X, mu_Y, sigma = args
    diff = norm([X - mu_X, Y - mu_Y])

    res = gaussian(X, Y, *args) * ((diff * diff) / (sigma ** 3) - 2 / sigma)

    return res


@jit
def partialmu_Xgaussian(X: int, Y: int, *args: float) -> float:
    alpha, mu_X, _, sigma = args

    res = gaussian(X, Y, *args) * (X - mu_X) / (sigma * sigma)

    return res


@jit
def partialmu_Ygaussian(X: int, Y: int, *args: float) -> float:
    alpha, _, mu_Y, sigma = args

    res = gaussian(X, Y, *args) * (Y - mu_Y) / (sigma * sigma)

    return res


_partials = [partialalphagaussian, partialmu_Xgaussian, partialmu_Ygaussian, partialsigmagaussian]


def fwhm_gaussian(*args: float):
    return 2 * sqrt(2 * log(2)) * args[-1]


def get_amplitude(X: int, Y: int, *args) -> float:
    """ Evaluate the Gaussian given some parameters at 0 """
    return args[0] / (2 * pi * args[-1])


class GD:
    """ A callable to perform GD """
    _counter: int = 0

    def __init__(self, model: Function, partials: List[Function]) -> None:
        self.model = model
        self.partials = partials

    def __call__(self, data: np.ndarray, learning_rate: float =1.0,
                 steps: int =1000, db: bool =True) -> List[float]:
        """ `Learn` the parameters of best fit for the given data and model """

        _min = data.min()
        _max = data.max()

        # scale amplitude to [0, 1]
        self.data = (data - _min) / (_max - _min)

        self.cubeX, self.cubeY = data.shape
        self.learning_rate = learning_rate
        self.steps = steps

        # perform the fit
        result = self.simplefit()

        # unscale amplitude of resultant
        result[0] = result[0] * (_max - _min) + _min

        result_as_list = result.tolist()

        self._counter += 1

        return result_as_list

    def simplefit(self) -> np.ndarray:
        """ Perform linear gradient descent """

        # Determine the center of mass of the star for a rough starting position
        # on mu_X and mu_Y
        com_X, com_Y = sum(np.array([i, j]) * self.data[i][j] for i, j in
                    product(range(self.cubeX), range(self.cubeY))) / np.sum(self.data)

        if not (0 <= com_X <= self.cubeX and 0 <= com_Y <= self.cubeY):
            warn(f'** star {self._counter} centroid lies outside boundaries')
            com_X, com_Y = self.cubeX // 2, self.cubeY // 2

        # initialize parameters
        parameters = np.array([random(), com_X, com_Y, (com_X + com_Y) / 4])

        # train the parameters using gradient descent
        for _ in range(self.steps):
            cost = self.cost(*parameters)

            cost[0] *= 150
            cost[3] *= 4.5

            parameters -= self.learning_rate * cost

        return parameters

    def cost(self, *args) -> np.ndarray:
        """ Get the cost of applying our model to the data as-is """
        n_args = len(args)
        total_cost = np.zeros(n_args)
        nabla_args = np.empty((n_args,))

        for i in range(self.cubeX):
            for j in range(self.cubeY):
                cost = self.model(i, j, *args) - self.data[i][j]

                for k in range(n_args):
                    nabla_args[k] = self.partials[k](i, j, *args)

                total_cost += cost * nabla_args

        normalized = total_cost / (self.cubeX * self.cubeY)

        return normalized


def generate_random_psf() -> None:
    """ Generate a random PSF to fit """
    _psf = Kernel(dims=(14, 14))
    psf = _psf.gaussian(sigma=2)
    psf += np.random.randn(14, 14) * 0.001
    psf *= 10000
    fits.writeto(os.getcwd() + '/random_psf.fits', psf, overwrite=True)



def test_gd() -> None:

    generate_random_psf()

    image_path_and_name = os.getcwd() + '/random_psf.fits'
    center = 7, 7

    base = os.path.splitext(os.path.basename(image_path_and_name))[0]

    bb = 7
    _Y = center[0] - bb, center[0] + bb
    _X = center[1] - bb, center[1] + bb
    unit = slice(*_X), slice(*_Y)

    with fits.open(image_path_and_name) as image:
        data = image[0].data

        # subtract background
        bkd = SEbkd().calc_background(data)
        data -= bkd

    star1 = data[unit]

    gd = GD(gaussian, _partials)

    from time import time

    _start = time()
    parameters = gd(star1, learning_rate=8.0, steps=60)
    print(f'** learning took {time() - _start}s')
    print(*parameters)
    print(f'** FWHM: {fwhm_gaussian(*parameters)}')

    # Plot the figure
    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    alpha, mu_X, mu_Y, sigma = parameters

    # ellipses at 1, 2 and 3 sd
    ellipses = [Ellipse(xy=(mu_X+0.5, mu_Y+0.5), height=2 * sigma * i, width=2 * sigma * i)
                for i in range(1, 4)]

    for e in ellipses:
        ax.add_artist(e)
        e.set_alpha(0.8)
        e.set_facecolor('none')
        e.set_edgecolor('b')

    fonts = {'fontsize': '8'}

    cax = ax.imshow(star1, origin='lower', cmap='gray', interpolation='none', extent=[0, 2 * bb, 0, 2 * bb])
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=fonts['fontsize'])
    cbar.set_label('ADU', size=fonts['fontsize'], rotation=270)

    ax.set_title(f'{center}    FWHM {str(fwhm_gaussian(*parameters))[:4]}    {base}', **fonts)
    ax.set_ylabel('Y (pix)', **fonts)
    ax.set_xlabel('X (pix)', **fonts)
    plt.setp(ax.get_xticklabels(), **fonts)
    plt.setp(ax.get_yticklabels(), **fonts)
    plt.tight_layout()
    #plt.show()

    plt.savefig('random_fit.png')


if __name__ == '__main__':
    test_gd()
