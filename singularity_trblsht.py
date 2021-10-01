# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from gridgraph.utils import param_loader


Jsol = 1
R_grid = 1
R = 1
E = 1e-12

Regularize = 1e-9               # Small factor


P_grid_at = lambda r: (Jsol * np.pi * (R**2 - r**2))**2 * R_grid / (2 * np.pi * r)


Es = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
Ps = []
for E in Es:
    Ps.append(quad(P_grid_at, E, R)[0])  # [W]


def simulate_iso(params, E):
    # Set some local vars for brevity, we need it where we're going...
    R = params['R']
    Jsol = params['Jsol']
    Voc = params['Voc']
    Rsheet = params['Rsheet']
    w_min = params['w_min']
    b_min = params['b_min']
    p_metal = params['Pwire'] / params['h']  # ~0.1 Ohm/sq

    # Helper equations
    C_at = lambda r: 2 * np.pi * r  # [cm] circumference
    # [A] current produced from edge to r:
    I_cumulative = lambda r: Jsol * np.pi * (R**2 - r**2)
    # [W] ideal maximum power from the circular cell
    P_max = Jsol * Voc * np.pi * R**2

    def P_grid(w, b, E):
        """
        Choose an optimal pitch 'b' and return the power LOSS from combined
        shadow, sheet, and line drop.
        Input: w width of the grid's metal lines
        Return: watts of power lost from all shadow, sheet, and line drop in
        the isotropic grid."""
        nonlocal P_max, R, Rsheet, p_metal

        # E = 1e-12               # Safety factor to avoid div-by-zero

        # [W] Power lost in sheet is not a function of w:
        # P_sheet = Jsol**2 * (np.pi / 32) * p_metal * R**2 * b**2
        P_sheet = Jsol**2 * (np.pi / 32) * Rsheet * R**2 * b**2
        # [W] Power lost due to shadowing
        P_shadow = 2 * P_max * w / (b + E)

        """ [W] Power lost in the grid lines """
        # [Ohms/sq.] Effective sheet resistance of the grid
        R_grid = p_metal * b / (w + E)
        # [W/cm] Power lost in a constant-radius slice of the circle
        P_grid_at = lambda r: I_cumulative(r)**2 * R_grid / C_at(r + E)
        P_grid = quad(P_grid_at, 0, R)[0]  # [W]

        # Regularizer: motivate large pitch
        Reg = -Regularize * b**2

        # [W] Power lost from all sources in the grid lines
        return P_sheet + P_shadow + P_grid + Reg

    # Find the optimal grid to minimize power loss
    optim = minimize(lambda x: P_grid(x[0], x[1], E),
                     x0=[R * 0.01, b_min],
                     bounds=((w_min, R * 0.5),
                             (b_min, R * 0.5)),
                     tol=1e-6)

    Power = P_max - optim.fun
    w, b = optim.x              # [cm] best width and pitch
    return Power, w, b


E = 1e-12
params = param_loader('./recipes/center_circle.csv')

powers = []
ws = []
bs = []
ratios = []

Es = np.logspace(-12, -6, num=13)
for E in Es:
    power_iso, w_iso, b_iso = simulate_iso(params, E)
    powers.append(power_iso)
    ws.append(w_iso)
    bs.append(b_iso)
    ratios.append(b_iso/w_iso)

for trend in ('powers', 'ws', 'bs', 'ratios'):
    plt.semilogx(Es, eval(trend))
    plt.title(trend)
    plt.xlabel('low limit of integration')
    plt.show()
