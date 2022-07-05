'''Support for numerical optimization/simulation of circle cells with a single
center sink.

Oliver Nakano-Baker
Jan. 2021'''

import logging

import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize

Regularize = 1e-9               # Small factor


def sweep_func_over(func, low, high, filename=None):
    '''Plot a function over 200 points in the range (low, high). Optionally
    save to disc location filename.'''
    STEPS = 200
    x = np.arange(low, high, (high - low) / STEPS)
    y = [func(a) for a in x]
    plt.plot(x, y)
    plt.ylim((0, np.max(y) * 1.1))
    if filename is None:
        plt.show()
    else:
        plt.save(filename)


def simulate_H(params, force=None):
    """Numerical simulation of the power output of a circular solar cell with
    H-bar grid design and optimal line thickness and pitch under the
    provided params.

    params - dictionary of material and simulation parameters
    force - optional tuple containing (width, pitch). If provided, no
    optimization is performed, instead the power output is estimated at the
    provided grid geometry.

    returns - Power [W], width[cm], pitch[cm]"""

    # Set some local vars for brevity, we need it where we're going...
    R = params['R']
    Jsol = params['Jsol']
    Voc = params['Voc']
    Rsheet = params['Rsheet']
    w_min = params['w_min']
    b_min = params['b_min']
    p_metal = params['Pwire'] / params['h']  # ~0.1 Ohm/sq
    E = params['epsilon']                    # cm

    ''' Half-circle functions '''
    # Length bus-to-edge as you travel out from the center sink
    l = lambda x: np.sqrt(R**2 - x**2)

    # Busbar trends:
    J_bus = lambda r: 2 * Jsol * l(r)      # [A/cm] incident current at x
    I_bus = lambda x: quad(J_bus, x, R)[0]  # [A] cumulative current in bus.
    P_bus = lambda x, w: I_bus(x)**2 * p_metal / w  # [W/cm] along 1 busbar

    """ Full-circle quantities & functions """
    P_max = Jsol * Voc * np.pi * R**2  # 100% FF [Watts]

    # Grid values:
    def P_grid(w, b):
        '''Input: w width of a grid's metal line
        Return: Power lost from shadow, sheet, and line drop in the grid of
        an H-Bar circular cell assuming optimal pitch.'''
        nonlocal P_bus, P_max, R, Rsheet, l, p_metal, E

        wbus = w                # bus and line widths equal

        # Busbar values:
        # [W] Total resistive loss in 2 bus bars:
        P_bus_line = 2 * quad(lambda x: P_bus(x, wbus), E, R)[0]
        # [W] Total lost in shadow for 2 bus bars:
        P_bus_shadow = 2 * R * Jsol * Voc * wbus

        # [W] Power lost due to grid shadowing
        P_shadow = P_max * w / b
        # [W] lost in the sheet (before finding a gridline) over the whole cell
        P_sheet = (np.pi / 12) * Jsol**2 * Rsheet * R**2 * b**2

        """Power lost in the grid in [W/cm] above or below the bus at a
        position x from the center along the bus."""
        def P_line_at(x):
            nonlocal w, b
            return (1 / 3) * Jsol**2 * p_metal * (b / w) * l(x)**3

        # [W] lost in the grid lines, in all 4 quadrants of the circular cell
        P_lines = 4 * quad(P_line_at, 0, R)[0]

        # Regularizer: motivate large pitch
        Reg = -Regularize * b**2

        return P_bus_line + P_bus_shadow + P_lines + P_shadow + P_sheet + Reg

    if force is not None:          # No solve, just run at the force geometry
        Power = P_max - P_grid(force[0], force[1])
        return Power, force[0], force[1]  # Power, w, b
        
    optim = minimize(lambda x: P_grid(x[0], x[1]),  # w, b
                 x0=[R * 0.01, b_min],
                 bounds=((w_min, R * 0.5),
                         (b_min, R * 0.5)),
                 tol=1e-6)
    Power = P_max - optim.fun
    w, b = optim.x
    return Power, w, b


def simulate_iso(params, force=None):
    """Numerical simulation of the power output of a circular solar cell with
    isotropic grid design and optimal line thickness and pitch under the
    provided params.

    params - dictionary of material and simulation parameters
    force - optional tuple containing (width, pitch). If provided, no
    optimization is performed, instead the power output is estimated at the
    provided grid geometry.

    returns - Power [W], width[cm], pitch[cm]"""
    # Set some local vars for brevity, we need it where we're going...
    R = params['R']
    Jsol = params['Jsol']
    Voc = params['Voc']
    Rsheet = params['Rsheet']
    w_min = params['w_min']
    b_min = params['b_min']
    p_metal = params['Pwire'] / params['h']  # ~0.1 Ohm/sq
    E = params['epsilon']                    # cm

    # Helper equations
    C_at = lambda r: 2 * np.pi * r  # [cm] circumference
    # [A] current produced from edge to r:
    I_cumulative = lambda r: Jsol * np.pi * (R**2 - r**2)
    # [W] ideal maximum power from the circular cell
    P_max = Jsol * Voc * np.pi * R**2

    def P_grid(w, b):
        """
        Choose an optimal pitch 'b' and return the power LOSS from combined
        shadow, sheet, and line drop.
        Input: w width of the grid's metal lines
        Return: watts of power lost from all shadow, sheet, and line drop in
        the isotropic grid."""
        nonlocal P_max, R, Rsheet, p_metal

        # [W] Power lost in sheet is not a function of w:
        P_sheet = Jsol**2 * (np.pi / 32) * Rsheet * R**2 * b**2
        # [W] Power lost due to shadowing
        P_shadow = 2 * P_max * w / b

        """ [W] Power lost in the grid lines """
        # [Ohms/sq.] Effective sheet resistance of the grid
        R_grid = p_metal * b / w
        # [W/cm] Power lost in a constant-radius slice of the circle
        P_grid_at = lambda r: I_cumulative(r)**2 * R_grid / C_at(r)
        P_grid = quad(P_grid_at, E, R)[0]  # [W]

        # Regularizer: motivate large pitch
        Reg = -Regularize * b**2

        # [W] Power lost from all sources in the grid lines
        return P_sheet + P_shadow + P_grid + Reg

    if force is not None:          # No solve, just run at the force geometry
        Power = P_max - P_grid(force[0], force[1])
        return Power, force[0], force[1]  # Power, w, b

    # Find the optimal grid to minimize power loss
    optim = minimize(lambda x: P_grid(x[0], x[1]),
                     x0=[R * 0.01, b_min],
                     bounds=((w_min, R * 0.5),
                             (b_min, R * 0.5)),
                     tol=1e-6)

    Power = P_max - optim.fun
    w, b = optim.x              # [cm] best width and pitch
    return Power, w, b
