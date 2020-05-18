import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from train_model import sample_likelihood, run_model, logistic_function
from parameters import defaults, scan_range, digits
from load_data import load_jhu, load_rki
from plotting import add_days, fit_quality, plot_model, plot_scan_pars, with_latex


parser = argparse.ArgumentParser()
parser.add_argument('-data_id', '--data_id', default='rki', type=str, help='Which data to use (rki, jhu).')
parser.add_argument('-discard_days', '--discard_days', default=0, type=int, help='Discard last days.')
parser.add_argument('-rmin', '--rmin', default=5., type=float, help='Minimum ellipse extension.')
parser.add_argument('-loss', '--loss', default='chi2', type=str, help='Fit to cumulative distribution.')
parser.add_argument('-ls', '--latex_style', default=False, action='store_true', help='Turn on Latex style plots.')
kw = parser.parse_args()

if kw.latex_style:
    mpl.rcParams.update(with_latex)

identifier = '_%s_loss%s' % (kw.data_id, kw.loss)
os.makedirs('img', exist_ok=True)
if kw.data_id == "rki":
    data = load_rki().iloc[52:-kw.discard_days] if kw.discard_days > 0 else load_rki().iloc[37:]
elif kw.data_id == "jhu":
    data = load_jhu().iloc[39:-kw.discard_days] if kw.discard_days > 0 else load_jhu().iloc[39:]
else:
    raise NotImplementedError("Data key %s not implemented!" % kw.data_id)
confirmed_data, dead_data = data.iloc[:, 0].to_numpy().astype(int) - 16, data.iloc[:, 1].to_numpy().astype(int)
days_data = data.index
confirmed_day_data, dead_day_data = np.diff(confirmed_data), np.diff(dead_data)
days = np.arange(len(confirmed_data))
print('Fitting to days: \n', days_data)
try:
    print('\nday-action-1: ', days_data[defaults['day-action-1']])
    print('day-action-2: ', days_data[defaults['day-action-2']])
except IndexError:
    pass

# scan_pars = ['burn-in', 'infected-start', 'detection-rate']
# scan_pars = ['lethality', 'detection-rate', 'infected-start']
# scan_pars = ['detection-rate', 'burn-in', 'R0-1']
# scan_pars = ['t-death', 'burn-in', 'R0-0']
# scan_pars = ['detection-rate', 'burn-in', 'R0-0']
# scan_pars = ['lethality', 'R0-0', 'R0-1']
# scan_pars = ['lethality', 'R0-1', 'R0-2']
# scan_pars = ['R0-lo-A', 'R0-lo-B', 'R0-lo-C']
# scan_pars = ['infected-start', 'detection-rate', 'R0-0']
# scan_pars = ['R0-0', 'R0-1', 'detection-rate']
# scan_pars = ['R0-0', 'R0-1', 'day-action-1']
scan_pars = ['R0-0', 'R0-1', 'R0-2']

lowest_like = np.inf
likelihoods = np.zeros([len(scan_range[key]) for key in scan_pars])
pars = {}
for i, a in enumerate(scan_range[scan_pars[0]]):
    print('%s: %s' % (scan_pars[0], a))
    for j, b in enumerate(scan_range[scan_pars[1]]):
        for k, c in enumerate(scan_range[scan_pars[2]]):
            pars.update({scan_pars[0]: a, scan_pars[1]: b, scan_pars[2]: c})
            like = sample_likelihood(pars, confirmed_data, dead_data, loss=kw.loss)
            likelihoods[i, j, k] = like
            if like < lowest_like:
                lowest_like = like
                min_idx = (i, j, k)

plot_scan_pars(scan_pars, likelihoods)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('img/likelihood%s.png' % identifier, bbox_inches='tight')
plt.close()

i, j, k = min_idx[0], min_idx[1], min_idx[2]
pars_opt = {scan_pars[_i]: scan_range[scan_pars[_i]][min_idx[_i]] for _i in range(len(scan_pars))}
print('\nBest parameters:')
for par in pars_opt:
    print('%s: %s' % (par, np.round(pars_opt[par], digits[par])))

# Create plot to show fit quality
cases, confirmed, dead, active = run_model(pars_opt, days.size)
fit_quality(days, confirmed_data, confirmed, dead_data, dead)
plt.savefig('img/fit_model%s.png' % identifier, bbox_inches='tight')
plt.close("all")


pred_len = 120 if (('R0-1' in scan_pars) or ('R0-lo-A' in scan_pars)) else 3
cases, confirmed, dead, active = run_model(pars_opt, days.size + pred_len)

fig, axs = plot_model(days_data, confirmed_data, confirmed, dead_data, dead, cases=cases, active=active, cut_data=False)
plt.tight_layout()
plt.savefig('img/predict_model_%s.png' % identifier, bbox_inches='tight')
plt.close()


if ('R0-lo-A' in scan_pars):
    days_pred = add_days(days_data, len(cases)-len(days_data))
    _R0 = logistic_function(np.arange(len(days_pred)), pars_opt.get('R0-0', defaults['R0-0']), pars_opt.get('R0-lo-A'),
                            pars_opt.get('R0-lo-B'), pars_opt.get('R0-lo-C'))
    plt.plot(days_pred, _R0, color='k')
    plt.xlabel("Time")
    plt.ylabel("R(0)")
    plt.grid(True)
    plt.savefig('img/reproduction_rate_%s.png' % identifier, bbox_inches='tight')
    plt.close()
