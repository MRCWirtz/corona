#!/usr/bin/env python
import numpy as np


def figsize(scale):
    # Get this from LaTeX using \the\textwidth
    fig_width_pt = 222
    # Convert pt to inch
    inches_per_pt = 1.0 / 72.27
    # Aesthetic ratio (you could change this)
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    # width in inches
    fig_width = fig_width_pt * inches_per_pt * scale
    # height in inches
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    return fig_size


with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.figsize": figsize(2),
    "legend.fancybox": False}
