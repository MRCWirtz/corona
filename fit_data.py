import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from train_model import sample_likelihood, run_model
from load_data import load_data
from plotting import with_latex
mpl.rcParams.update(with_latex)

scan_range = {'lethality': np.arange(0.005, 0.055, 0.005),
              'burn-in': np.arange(5, 20, 1),
              't-death': np.arange(14, 32, 2),
              't-confirmed': [7],
              'infected-start': np.arange(5, 55, 5),
              'detection-rate': np.arange(0.1, 0.9, 0.1),
              'R0-0': np.arange(2., 3.2, 0.1),
              'R0-1': np.arange(0.6, 3.2, 0.2)}
digits = {
    'lethality': 3,
    'burn-in': 0,
    't-death': 0,
    'detection-rate': 2,
    'R0-0': 1,
    'R0-1': 1,
    'infected-start': 0,
    't-confirmed': 0,
}


data = load_data()
os.makedirs('img', exist_ok=True)

confirmed_data, dead_data = data.to_numpy()[36:, 0] - 16, data.to_numpy()[36:, 1]
confirmed_day_data, dead_day_data = np.diff(confirmed_data), np.diff(dead_data)
days = np.arange(len(confirmed_data))
print(confirmed_data)

# scan_pars = ['burn-in', 'infected-start', 'R0-0']
# scan_pars = ['lethality', 'detection-rate', 'burn-in']
# scan_pars = ['detection-rate', 'burn-in', 'R0-1']
# scan_pars = ['t-death', 'burn-in', 'R0-0']
# scan_pars = ['detection-rate', 'burn-in', 'R0-0']
scan_pars = ['t-confirmed', 'R0-0', 'R0-1']
day_action = 17 if ('R0-1' in scan_pars) else None
# day_action = 17

lowest_like = np.inf
likelihoods = np.zeros([len(scan_range[key]) for key in scan_pars])
pars = {}
for i, a in enumerate(scan_range[scan_pars[0]]):
    print('%s: %s' % (scan_pars[0], a))
    for j, b in enumerate(scan_range[scan_pars[1]]):
        for k, c in enumerate(scan_range[scan_pars[2]]):
            pars.update({scan_pars[0]: a, scan_pars[1]: b, scan_pars[2]: c})
            like = sample_likelihood(pars, confirmed_day_data, dead_day_data, day_action=day_action)
            likelihoods[i, j, k] = like
            if like < lowest_like:
                lowest_like = like
                min_idx = (i, j, k)

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
plt.savefig('img/likelihood_{}.png'.format('_'.join(scan_pars)), bbox_inches='tight')

i, j, k = min_idx[0], min_idx[1], min_idx[2]
pars_opt = {scan_pars[_i]: scan_range[scan_pars[_i]][min_idx[_i]] for _i in range(len(scan_pars))}
print('\nBest parameters:')
for par in pars_opt:
    print('%s: %s' % (par, np.round(pars_opt[par], digits[par])))

pred_len = 42 if ('R0-1' in scan_pars) else 3
burn_in = pars_opt['burn-in'] if ('burn-in' in scan_pars) else 9
days_sim = days.size + burn_in + pred_len

pars_opt.update({"R0-2": 0.8})
cases, confirmed, dead = run_model(pars_opt, days_sim, n_burn_in=burn_in, day_action=day_action)

fig, axs = plt.subplots(2, 1)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax in axs:
    ax.set_xlabel("days")

axs[0].scatter(burn_in + days, confirmed_data, marker='o', color='k', label='data (germany)')
axs[0].plot(np.arange(days_sim)[:-pred_len], confirmed[:-pred_len], color='k', label='model')
axs[0].legend(loc="upper left")
axs[0].set_ylabel("Confirmed Cases")
axs[0].set_yscale("log")
axs[0].set_ylim([0.1, 1.5*np.max(confirmed_data)])
axs[0].legend(loc='upper left', fontsize=14)

axs[1].scatter(burn_in + days, dead_data, marker='o', color='k', label='data (germany)')
axs[1].plot(np.arange(days_sim)[:-pred_len], dead[:-pred_len], color='red', label='model')
axs[1].legend(loc="upper left")
axs[1].set_ylabel("Dead")
axs[1].set_yscale("log")
axs[1].set_ylim([0.01, 1.5*np.max(dead_data)])
axs[1].legend(loc='upper left', fontsize=14)
plt.savefig('img/fit_model_%s' % '_'.join(scan_pars), bbox_inches='tight')
plt.close("all")


# Prediction
fig, axs = plt.subplots(2, 2)
fig.set_figheight(9)
fig.set_figwidth(16)

cases, confirmed, dead = cases[burn_in:], confirmed[burn_in:], dead[burn_in:]
days_pred = np.arange(len(cases))

axs[0, 0].plot(days_pred, cases, color='blue', label='total (model)')
axs[0, 0].plot(days_pred, confirmed, color='k', label='confirmed (model)')
axs[0, 0].scatter(days, confirmed_data, marker='o', color='k', label='data (Germany)')
axs[0, 0].set_ylabel("Cases")
axs[0, 0].legend(loc='upper left', fontsize=14)

axs[1, 0].plot(days_pred, dead, color='r', label='model')
axs[1, 0].scatter(days, dead_data, marker='o', color='k', label='data (Germany)')
axs[1, 0].set_xlabel("Days")
axs[1, 0].set_ylabel("Deaths")
axs[1, 0].legend(loc='upper left', fontsize=14)

axs[0, 1].plot(days_pred[1:], np.diff(cases), color='b', ls="dashed", label='total (model)')
axs[0, 1].plot(days_pred[1:], np.diff(confirmed), color='k', ls="dashed", label='confirmed (model)')
axs[0, 1].scatter(days[1:], confirmed_day_data, marker='o', color='k', label='data (Germany)')
axs[0, 1].set_ylabel("New cases per day")
axs[0, 1].legend(loc='upper left', fontsize=14)

axs[1, 1].plot(days_pred[1:], np.diff(dead), color='r', ls="dashed", label='model')
axs[1, 1].scatter(days[1:], dead_day_data, marker='o', color='k', label='data (Germany)')
axs[1, 1].legend(loc="upper left")
axs[1, 1].set_xlabel("Days")
axs[1, 1].set_ylabel("New deaths per day")
axs[1, 1].legend(loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig('img/predict_model_%s.png' % '_'.join(scan_pars), bbox_inches='tight')
plt.close()
