#!/usr/bin/env python
import numpy as np
from copy import copy
from matplotlib import dates
import matplotlib.pyplot as plt
from datetime import timedelta

from parameters import scan_range, digits


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


def add_days(days, add_days=0):
    """ Add number of abs(add_days) days to existing datetime Object, if negative in the beginning """
    days = copy(days)
    for i in range(abs(add_days)):
        idx = 0 if add_days < 0 else len(days)
        days = days.insert(idx, days[idx-1] + np.sign(add_days)*timedelta(days=1))
    return days


def fit_quality(days, confirmed_data, confirmed, dead_data, dead):

    fig, axs = plt.subplots(2, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    for ax in axs:
        ax.set_xlabel("days")

    axs[0].scatter(days, confirmed_data, marker='o', color='k', label='data (germany)')
    axs[0].plot(np.arange(days.size), confirmed, color='k', label='model')
    axs[0].legend()
    axs[0].set_ylabel("Confirmed Cases")
    axs[0].set_yscale("log")
    axs[0].set_ylim([0.1, 1.5*np.max(confirmed_data)])
    axs[0].legend()

    axs[1].scatter(days, dead_data, marker='o', color='k', label='data (germany)')
    axs[1].plot(np.arange(days.size), dead, color='red', label='model')
    axs[1].legend()
    axs[1].set_ylabel("Dead")
    axs[1].set_yscale("log")
    axs[1].set_ylim([0.01, 1.5*np.max(dead_data)])
    axs[1].legend()

    return fig, axs


def plot_model(days_data, confirmed_data, confirmed, dead_data, dead, cases=None, active=None, cut_data=False):

    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(9)
    fig.set_figwidth(16)

    days_pred = add_days(days_data, len(cases)-len(days_data))
    axs[0, 0].plot_date(days_pred, confirmed, marker='None', ls='solid', color='k', label=r'confirmed (model)')
    if cases is not None:
        axs[0, 0].plot_date(days_pred, cases, marker='None', ls='solid', color='b', label=r'total (model)')
    if active is not None:
        axs[0, 0].plot_date(days_pred, active, marker='None', ls='solid', color='g', label=r'active (model)')
    axs[0, 1].plot_date(days_pred[1:], np.diff(confirmed), ls='solid', marker='None', color='k', label=r'Model')
    axs[1, 0].plot_date(days_pred, dead, marker='None', ls='solid', color='r', label=r'Model')
    axs[1, 1].plot_date(days_pred[1:], np.diff(dead), marker='None', ls='solid', color='r', label=r'Model')

    axs[0, 0].xaxis.set_minor_locator(dates.DayLocator())
    axs[0, 0].xaxis.set_major_locator(dates.MonthLocator())
    axs[0, 0].xaxis.set_major_formatter(dates.DateFormatter('%b'))
    axs[0, 0].plot_date(days_data, confirmed_data, marker='o', color='k', label='Data (Germany)')
    axs[0, 0].set_ylabel("Cases")
    if cut_data:
        axs[0, 0].set_ylim(0, 1.5*np.max(confirmed_data))
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].xaxis.set_minor_locator(dates.DayLocator())
    axs[0, 1].xaxis.set_major_locator(dates.MonthLocator())
    axs[0, 1].xaxis.set_major_formatter(dates.DateFormatter('%b'))
    axs[0, 1].plot_date(days_data[1:], np.diff(confirmed_data), marker='o', color='k', label='Data (Germany)')
    axs[0, 1].set_ylabel("New cases per day")
    if cut_data:
        axs[0, 1].set_ylim(0, 1.5*np.max(np.diff(confirmed_data)))
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].xaxis.set_minor_locator(dates.DayLocator())
    axs[1, 0].xaxis.set_major_locator(dates.MonthLocator())
    axs[1, 0].xaxis.set_major_formatter(dates.DateFormatter('%b'))
    axs[1, 0].plot_date(days_data, dead_data, marker='o', color='k', label='Data (Germany)')
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Deaths")
    if cut_data:
        axs[1, 0].set_ylim(0, 1.5*np.max(dead_data))
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].xaxis.set_minor_locator(dates.DayLocator())
    axs[1, 1].xaxis.set_major_locator(dates.MonthLocator())
    axs[1, 1].xaxis.set_major_formatter(dates.DateFormatter('%b'))
    axs[1, 1].plot_date(days_data[1:], np.diff(dead_data), marker='o', color='k', label='Data (Germany)')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("New deaths per day")
    if cut_data:
        axs[1, 1].set_ylim(0, 1.5*np.max(np.diff(dead_data)))
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    return fig, axs


def plot_scan_pars(scan_pars, likelihoods):

    x = np.array([0, 1, 2])
    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    idx = 0
    for i, par_i in enumerate(scan_pars):
        for j, par_j in enumerate(scan_pars):
            if j <= i:
                continue
            mask = (x != i) & (x != j)
            like_proj = np.min(likelihoods, axis=np.where(mask)[0][0])
            plt.sca(axs[idx])
            idx += 1
            plt.imshow(like_proj.T, cmap='inferno_r')
            cbar = plt.colorbar()
            cbar.set_label('Deviance')
            if len(scan_range[par_i]) > 1:
                xticks = np.rint(np.linspace(0, len(scan_range[par_i])-1, min(6, len(scan_range[par_i])))).astype(int)
                plt.xticks(xticks, np.round(scan_range[par_i][xticks], digits[par_i]))
            if len(scan_range[par_j]) > 1:
                yticks = np.rint(np.linspace(0, len(scan_range[par_j])-1, min(6, len(scan_range[par_j])))).astype(int)
                plt.yticks(yticks, np.round(scan_range[par_j][yticks],  digits[par_j]))
            plt.xlabel('%s' % par_i)
            plt.ylabel('%s' % par_j)

    return fig
