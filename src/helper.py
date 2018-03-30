#!/usr/bin/env python3
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# this script studies the influence 
# of Apz Br2 and omega on the motion
# of the leaf
##################################

import numpy as np
import matplotlib.pyplot as plt


def Br2_phi():
    Br2s = [5, 10, 20, 40, 80, 100, 200, 500, 800, 1000, 2000]
    Apz = 0.01
    omega = np.pi / 100
    min_phis = [22.65, 28.44, 36.39, 47.11, 61.49, 67.09, \
            88.42, 129.26, 158.46, 174.89, 235.14]
    plt.plot(Br2s, min_phis, 'b-o', linewidth=2)
    plt.title(r'$Br^2$ vs $\phi_{min}$ (Apz = 0.01, $\omega$ = 0.01 $\pi$)')
    plt.xlabel(r'$Br^{2}$')
    plt.ylabel(r'$\phi_{min}$')


def Br2_energy():
    Br2s = [5, 10, 20, 40, 80, 100, 200, 500, 800, 1000, 2000]
    es = [0.1836, 0.0889, 0.0426, 0.0201, 0.0093, 0.0072, \
            0.0032, 0.0010, 0.0005, 0.0003, 0.0001]
    plt.plot(Br2s, es, 'g-o', linewidth=2)
    plt.title(r'$Br^2$ vs energy (Apz = 0.01, $\omega$ = 0.01 $\pi$)')
    plt.xlabel(r'$Br^{2}$')
    plt.ylabel('energy harvested in one period')


def Apz_phi():
    Apzs = [0.01, 0.02, 0.05, 0.1, 0.5, 1]
    phis = [67.10, 67.61, 69.10, 71.47, 86.80, 101.03]
    plt.plot(Apzs, phis, 'b-o', linewidth=2)
    plt.title(r'Apz vs $\phi_{min}$ (Br2 = 100, $\omega$ = 0.01 $\pi$)')
    plt.xlabel(r'Apz')
    plt.ylabel(r'$\phi_{min}$')


def Apz_energy():
    Apzs = [0.01, 0.02, 0.05, 0.1, 0.5, 1]
    es = [0.0072, 0.0144, 0.0360, 0.0720, 0.3833, 0.7341]
    plt.plot(Apzs, es, 'g-o', linewidth=2)
    plt.title(r'Apz vs energy (Br2 = 100, $\omega$ = 0.01 $\pi$)')
    plt.xlabel(r'Apz')
    plt.ylabel('energy harvested in one period')


def omega_phi():
    omegas = np.pi * np.array([1/5, 1/10, 1/20, 1/50, 1/80, 1/100, 1/120])
    Br2 = 100
    Apz = 0.01
    phis = [344.88, 329.77, 299.54, 208.84, 119.41, 67.10, 32.86]
    plt.plot(omegas, phis, 'b-o', linewidth=2)
    plt.title(r'$\omega$ vs $\phi_{min}$ (Apz = 0.01, Br2 = 100)')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\phi_{min}$')


def omega_energy():
    omegas = np.pi * np.array([1/5, 1/10, 1/20, 1/50, 1/80, 1/100, 1/120])
    Br2 = 100
    Apz = 0.01
    es = [0.00037, 0.00074, 0.00148, 0.00370, 0.00590, 0.0072, 0.0081]
    plt.plot(omegas, es, 'g-o', linewidth=2)
    plt.title(r'$\omega$ vs energy (Apz = 0.01, Br2 = 100)')
    plt.xlabel(r'$\omega$')
    plt.ylabel('energy harvested in one period')


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    # Br2_phi()
    # Br2_energy()
    # Apz_phi()
    # Apz_energy()
    omega_phi()
    # omega_energy()
    plt.show()

