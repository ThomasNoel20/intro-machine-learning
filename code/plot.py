"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FormatStrFormatter 

def make_cmaps():
    """
    Return
    ------
    bg_map, sc_map: tuple (colormap, colormap)
        bg_map: The colormap for the background
        sc_map: Binary colormap for scatter points
    """
    # Class 0 is blue and class 1 is orange, as in the dataset figure
    # produced by data.py. The background shows P(y=1|x).
    top = mpl.colormaps['Blues_r']
    bottom = mpl.colormaps['Oranges']

    newcolors = np.vstack((top(np.linspace(.25, 1., 128)),
                           bottom(np.linspace(0., .75, 128))))
    bg_map = ListedColormap(newcolors, name='BlueOrange')

    sc_map = ListedColormap(['DodgerBlue', '#ff8000'])

    return bg_map, sc_map

def plot_boundary(fname, fitted_estimator, X, y, mesh_step_size=0.1, title=""):
    """Plot estimator decision boundary and scatter points

    Parameters
    ----------
    fname : str
        File name where the figures is saved.

    fitted_estimator : a fitted estimator

    X : array, shape (n_samples, 2)
        Input matrix

    y : array, shape (n_samples, )
        Binary classification target

    mesh_step_size : float, optional (default=0.1)
        Mesh size of the decision boundary

    title : str, optional (default="")
        Title of the graph

    """
    bg_map, sc_map = make_cmaps()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary.
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X_0")
    plt.ylabel("X_1")

    # Put the result into a color plot
    cf = plt.contourf(xx, yy, Z, cmap=bg_map, levels=11, alpha=.8, vmin=0, vmax=1)

    # Plot test points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=sc_map, edgecolor='black', s=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    ticks = np.arange(0.0, 1.1, 0.1)
    plt.clim(0.0, 1.0)
    cbar = plt.colorbar(cf, ticks=ticks)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])

    # Found no other way to add the missing tick.
    if np.min(Z) == np.max(Z):
        locator = FixedLocator([np.min(Z)])
        formatter = FormatStrFormatter('%g')
        cbar.locator = locator
        cbar.formatter = formatter
        cbar.update_ticks()

    plt.savefig('{}.pdf'.format(fname), transparent=True)
    plt.close()
