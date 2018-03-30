#!/usr/bin/env python3
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# This module do some testing
##################################


import numpy as np
import matplotlib.pyplot as plt


def compute_quadratic_root(a, b, c):
    """ compute roots of ax^2 + bx + c = 0 """
    temp = np.sqrt(b**2 - 4 * a * c)
    return (-b - temp) / (2 * a), (-b + temp) / (2 * a)


def test_latex_matplotlib():
    """ test latex in matplotlib """
    plt.rc('text', usetex=True)
    plt.plot(range(10))
    x = 1
    plt.title(r'$x^2 = {}$'.format(x))
    plt.show()


if __name__ == "__main__":
    # print(compute_quadratic_root(1, -6, 8))
    test_latex_matplotlib()

