#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple gradient descent class for fitting Gaussian and Moffat functions.

It has been written, however, to accept any surface and that function's first
order partial derivatives.

Also included is a function for retrieving data from FITS images using PyFITS.

For further details, see my blog post at

https://aperiodicity.com/2016/06/02/stellar-distributions/
"""

__author__ = 'Brandon Doyle'

import partials
from kernel import *
from numpy.linalg import norm
from numpy.random import randn
import numpy as np
import pyfits
import sys

class GD(object):
    def __init__(self, array, form, derivatives, max_iterations=1000, \
            limit=0.01):
        
        if (type(array).__module__ != np.__name__):
            # handle lists as well
            self.maximum = np.array(array).max()
            self.minimum = np.array(array).min()
        else:
            self.maximum = array.max()
            self.minimum = array.min()

        if (self.maximum != 1.0 or self.minimum != 0.0):
            # Map range of image array → [0, 1]
            self.array = np.array(map(lambda arr: \
                (arr-self.minimum)/(self.maximum-self.minimum), array))
        else:
            # probably not common in practice...
            self.array = np.array(array)
        
        if (self.dim(self.array) > 2):
            raise ValueError('Array must be dimension 2.')
        
        self.rangeX = self.array.shape[0]
        self.rangeY = self.array.shape[1]
        self.form = form
        self.max_iterations = max_iterations
        self.limit = limit
        self.derivs = derivatives

    @staticmethod
    def dim(arr):
        if (type(arr).__module__ != np.__name__):
            return len(np.array(arr).shape)
        else:
            return len(arr.shape)

    @staticmethod
    def expected(arr):
        if (type(arr).__module__ != np.__name__):
            return np.average(arr)
        else:
            return np.average(np.array(arr))

    def linear(self):
        """ perform linear GD using all derivatives """
        self.param_list = []
        costs = []

        # randomly generate initial parameter vector
        theta = np.abs(randn(len(self.derivs) + 1)) + \
            (((self.rangeX + self.rangeY) // 2) - 1) // 2
        theta[0] = 1.0

        # perform GD using all partial derivatives
        for _ in xrange(self.max_iterations):
            cost = self.__cost_(theta)
            theta += self.__gamma_(1.0)*cost

            self.param_list.append(theta.tolist())
            
            ### Conditions only added to watch outputs -- may have to customize these
            if (len(self.derivs) + 1 == 4):
                sys.stdout.write('%f\t%f\t%f\t%f\r' % (theta[0], theta[1], \
                                                 theta[2], theta[3]))
                sys.stdout.flush()
            elif (len(self.derivs) + 1 == 5):
                sys.stdout.write('%f\t%f\t%f\t%f\t%f\r' % (theta[0], theta[1], \
                                                   theta[2], theta[3], \
                                                   theta[4]))
                sys.stdout.flush()

            if (not (0 < theta[-2] < self.rangeX) or \
                not (0 < theta[-1] < self.rangeY)):
                break
            if (norm(cost) < self.limit):
                break
            if (norm(cost) > 10e3):
                print "Values diverged."
                sys.exit(0)

        return theta, np.array(self.param_list)

    def __gamma_(self, magnitude=1.0):
        """ estimate the value of γ """
        if (len(self.derivs) + 1 == 4):
            # tuned for a decent Gaussian GD
            return -magnitude*np.array([1e3, 25e1, 2e3, 2e3])
        elif (len(self.derivs) + 1 == 5):
            # tuned for a decent Moffat GD
            return -magnitude*np.array([5e2, 5e2, 5e1, 1e2, 1e2])

    def __cost_(self, theta):
        """ compute the cost, Ω(mN^2) time.

        Cost is a vector with the same dimensions as theta due to nabla. 
        """
        total_cost = np.zeros(len(self.derivs) + 1)

        for i in xrange(self.rangeX):
            for j in xrange(self.rangeY):
                # get difference between model and array
                cost = self.form([i, j], theta) - self.array[i][j]
                
                # set up nabla
                nabla = np.zeros(len(self.derivs) + 1)
                
                # compute partials at each pixel
                for k in range(len(self.derivs) - 1):
                    nabla[k] = self.derivs[k]([i, j], theta)
                else:
                    # compute the last derivative twice, once for each
                    # axis. This may be collapsed into one line
                    # with a slight modification to the partial 
                    # functions.
                    nabla[-1] = self.derivs[-1]([i, j], theta, NAXIS=1)
                    nabla[-2] = self.derivs[-1]([i, j], theta, NAXIS=0)

                # scale nabla by the difference
                total_cost += cost*nabla
        else:
            return total_cost / (self.rangeX*self.rangeY)

def get_fits_array(image_name, axis=0, info=False):
    """ Get the data from an image along a specified axis. """
    try:
        image = pyfits.open(image_name, mode='readonly')
        if (info):
            print image.info()
        return image[axis].data
    finally:
        image.close()
