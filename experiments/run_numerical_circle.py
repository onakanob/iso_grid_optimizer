'''Experiment to compare numerical models for 3 grid morphologies covering a
circular cell with a sink at the center.

Oliver Nakano-Baker
Jan. 2021'''

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append('..')
from gridgraph.utils import param_loader, start_logging
from gridgraph.numerical_circle_cell import simulate_H, simulate_iso


def simulate_H_and_iso(params):
    """Run H-bar and isometric circle patterns with the params provided.
    params - dictionary of simulation parameters
    results - pandas dataframe to which results should be appended"""
    try:
        R = params['R']
        power_H, w_H, b_H = simulate_H(params)
        power_iso, w_iso, b_iso = simulate_iso(params)
        return {**params,
                'H-Bar W': power_H,
                'H-Bar W/cm2': power_H / (np.pi * R**2),
                'H-Bar width': w_H,
                'H-Bar pitch': b_H,
    
                'Isotropic W': power_iso,
                'Isotropic W/cm2': power_iso / (np.pi * R**2),
                'Isotropic width': w_iso,
                'Isotropic pitch': b_iso}
    except:
        return {**params,
                'H-Bar W': None,
                'H-Bar W/cm2': None,
                'H-Bar width': None,
                'H-Bar pitch': None,
    
                'Isotropic W': None,
                'Isotropic W/cm2': None,
                'Isotropic width': None,
                'Isotropic pitch': None}


def vis_trend(df, xaxis, xlabel, title, directory, logscale=False):
    """Save a plot of H-bar vs Isotropic W/cm2 for the specified x-axis"""
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=[9, 7], dpi=300)

    if logscale:
        ax.semilogx(df[xaxis], df['H-Bar W/cm2'], 'k', linestyle='dotted')
        ax.semilogx(df[xaxis], df['Isotropic W/cm2'], 'k', linestyle='dashed')
    else:
        ax.plot(df[xaxis], df['H-Bar W/cm2'], 'k', linestyle='dotted')
        ax.plot(df[xaxis], df['Isotropic W/cm2'], 'k', linestyle='dashed')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('W/cm2')
    ax.legend(('H-Bar', 'Isotropic'))

    fig.savefig(os.path.join(directory, title + '.png'))


# TODO switched back to logspace sweeps
def compare_H_vs_Iso(params):
    RESOLUTION = 7
    output_csv = os.path.join(params['log_dir'], 'results.csv')
    results = pd.DataFrame()    # Final output

    logging.info('Metal sheet resistance: ' + str(params['Pwire'] /
                                                  params['h']) + ' Ohms/sq.')

    df = simulate_H_and_iso(params)
    results = results.append(df, ignore_index=True)
    logging.info('Center point: ' + str(df))

    # EXPERIMENT: Sweep Sheet Resistivity
    Rs = np.logspace(np.log10(1), np.log10(10000), num=RESOLUTION,
                     endpoint=True, base=10)
    # Rs = np.array([46.42, 215.44, 1000, 4641.59, 21544.35])
    df = pd.DataFrame()
    for Rsheet in Rs:
        logging.info('Optimizing grids for Rsheet = %.3f' % Rsheet)
        df = df.append(simulate_H_and_iso({**params, 'Rsheet': Rsheet}),
                       ignore_index=True)
    results = results.append(df)
    # Create and save a visualization of the points just calculated
    vis_trend(df=df,
              xaxis='Rsheet', xlabel='sheet resistance [Ohm/sq.]',
              title='Varying Sheet Resistance', directory=params['log_dir'],
              logscale=True)


    # EXPERIMENT: Sweep Metal Resistivity
    Ps = np.logspace(np.log10(10**-7.5), np.log10(10**-5.5), num=RESOLUTION,
                     endpoint=True, base=10)
    # Ps = np.array([6.81E-08, 1.47E-07, 3.16E-07, 6.81E-07, 1.00E-06, 1.47E-06])
    df = pd.DataFrame()
    for Pwire in Ps:
        logging.info('Optimizing grids for Pwire = %.3f' % Pwire)
        df = df.append(simulate_H_and_iso({**params, 'Pwire': Pwire}),
                       ignore_index=True)
    results = results.append(df)
    # Create and save a visualization of the points just calculated
    vis_trend(df=df,
              xaxis='Pwire', xlabel='metal resistivity [Ohm-cm]',
              title='Varying Metal Resistivity', directory=params['log_dir'],
              logscale=True)

    # EXPERIMENT: Sweep R
    Rs = np.logspace(-1, np.log10(20), num=RESOLUTION, endpoint=True, base=10)
    # Rs = np.array([0.242, 0.585, 1.414, 3.42, 5, 8.27])
    df = pd.DataFrame()    
    for R in Rs:
        logging.info('Optimizing grids for R = %.3f' % R)
        df = df.append(simulate_H_and_iso({**params, 'R': R}),
                       ignore_index=True)
    results = results.append(df)
    # Create and save a visualization of the points just calculated
    vis_trend(df=df,
              xaxis='R', xlabel='radius [cm]',
              title='Varying Cell Radius', directory=params['log_dir'],
              logscale=False)

    # EXPERIMENT: Sweep Jsol
    Js = np.logspace(np.log10(.0002), np.log10(0.2), num=RESOLUTION,
                     endpoint=True, base=10)
    # Js = np.array([0.000632, 0.002, 0.006325, 0.02, 0.063246])
    df = pd.DataFrame()
    for J in Js:
        logging.info('Optimizing grids for Jsol = %.3f' % J)
        df = df.append(simulate_H_and_iso({**params, 'Jsol': J}),
                       ignore_index=True)
    results = results.append(df)
    # Create and save a visualization of the points just calculated
    vis_trend(df=df,
              xaxis='Jsol', xlabel='solar current [A/cm2]',
              title='Varying Solar Current', directory=params['log_dir'],
              logscale=True)

    # After all experiments, log the run:
    results.to_csv(output_csv, index=False)
    logging.info('Run complete - results saved to %s', output_csv)


def wobble_about_optimal(params):
    """Get results adjacent to the optimum H-bar center point."""
    output_csv = os.path.join(params['log_dir'], 'wobble_about_optimal.csv')
    STEP = 0.85
    steps = np.power(STEP, np.arange(-4, 5))
    results = pd.DataFrame()

    def log_result(power, w, b):
        nonlocal results, params
        results = results.append({**params,
                                  'power': power,
                                  'width': w,
                                  'pitch': b},
                                 ignore_index=True)

    # TODO skip this force
    # power, best_w, best_b = simulate_H(params)
    best_w = 0.0213923763549842
    best_b = 0.17134455868565
    power, _, _ = simulate_H(params, force=(best_w, best_b))
    log_result(power, best_w, best_b)
    logging.info('Center optimal (power, width, pitch): ' + str((power, best_w, best_b)))

    # Pitches wobble
    bs = np.round(best_b * steps, decimals=3)
    for this_b in bs:
        power, w, b = simulate_H(params, force=(best_w, this_b))
        log_result(power, w, b)

    ws = np.round(best_w * steps, decimals=3)
    for this_w in ws:
        power, w, b = simulate_H(params, force=(this_w, best_b))
        log_result(power, w, b)

    results.to_csv(output_csv, index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='../recipes/center_circle.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)
    start_logging(args.log_dir, 'numerical circle experiment')

    compare_H_vs_Iso({**vars(args), **params})
    wobble_about_optimal({**vars(args), **params})
