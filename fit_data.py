import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from train_model import likelihood
from models import DayDrivenPandemie
from load_data import load_data
from latex_style import with_latex
mpl.rcParams.update(with_latex)

np.random.seed(1)

data = load_data()
os.makedirs('img', exist_ok=True)

confirmed_data, dead_data = data.to_numpy()[36:, 0] - 16, data.to_numpy()[36:, 1]
confirmed_day_data, dead_day_data = np.diff(confirmed_data), np.diff(dead_data)
days = np.arange(len(confirmed_data))


def sample_likelihood(pars):
    infected_confirmed, dead = np.zeros(days.size), np.zeros(days.size)
    world = DayDrivenPandemie(n_days=days.size,
                              n_p=pars[0]/0.15,
                              attack_rate=0.15,
                              detection_rate=0.8,
                              lethality=pars[1],
                              infected_start=int(pars[2]),
                              contagious_start=int(pars[2]*2/3),
                              confirmed_start=confirmed_data[0])

    for i in days:
        world.update()
        infected_confirmed[i], dead[i] = world.confirmed_total, world.dead

    return -likelihood(np.diff(infected_confirmed), confirmed_day_data, np.diff(dead), dead_day_data)


lowest_like = np.inf
r0_candidates = np.arange(2., 3.1, 0.1)
lethality_candidates = np.arange(0.005, 0.06, 0.005)
infected_start_candidates = np.arange(100, 500, 25)
likelihoods = np.zeros((len(r0_candidates), len(lethality_candidates), len(infected_start_candidates)))
for i, n in enumerate(r0_candidates):
    for j, l in enumerate(lethality_candidates):
        for k, i_s in enumerate(infected_start_candidates):
            like = sample_likelihood([n, l, i_s])
            print(n, l, i_s, like)
            likelihoods[i, j, k] = like
            if like < lowest_like:
                lowest_like = like
                min_idx = (i, j, k)

like_proj = np.min(likelihoods, axis=1)
plt.imshow(np.transpose(like_proj), cmap='inferno_r')
xticks = np.rint(np.linspace(0, len(r0_candidates)-1, 6)).astype(int)
plt.xticks(xticks, np.round(r0_candidates[xticks], 1))
yticks = np.rint(np.linspace(0, len(infected_start_candidates)-1, 6)).astype(int)
plt.yticks(yticks, np.round(infected_start_candidates[yticks], 1))
plt.xlabel("R0")
plt.ylabel("infected start")
plt.savefig('img/likelihood_r0_infected-start.png', bbox_inches='tight')
plt.close('all')

like_proj = np.min(likelihoods, axis=2)
plt.imshow(np.transpose(like_proj), cmap='inferno_r')
plt.xticks(xticks, np.round(r0_candidates[xticks], 1))
yticks = np.rint(np.linspace(0, len(lethality_candidates)-1, 6)).astype(int)
plt.yticks(yticks, np.round(lethality_candidates[yticks], 3))
plt.xlabel("np")
plt.ylabel("lethality")
plt.savefig('img/likelihood_r0_lethality.png', bbox_inches='tight')
plt.close('all')

like_proj = np.min(likelihoods, axis=2)
plt.imshow(np.transpose(like_proj), cmap='inferno_r')
xticks = np.rint(np.linspace(0, len(infected_start_candidates)-1, 6)).astype(int)
plt.xticks(yticks, np.round(infected_start_candidates[xticks], 1))
plt.yticks(yticks, np.round(lethality_candidates[yticks], 3))
plt.xlabel("np")
plt.ylabel("lethality")
plt.savefig('img/likelihood_infected-start_lethality.png', bbox_inches='tight')
plt.close('all')

i, j, k = min_idx[0], min_idx[1], min_idx[2]
print('\nBest parameters:')
print('R0 = %.1f' % r0_candidates[i])
print('infected start: %.1f' % infected_start_candidates[k])
print('lethality: %.4f' % lethality_candidates[j])
pred_len = 40
days_pred = np.arange(days.size + pred_len)
infected, infected_confirmed = np.zeros(days_pred.size), np.zeros(days_pred.size)
infected_day = np.zeros(days_pred.size)
cured, dead = np.zeros(days_pred.size), np.zeros(days_pred.size)
world = DayDrivenPandemie(n_days=days.size+pred_len,
                          n_p=r0_candidates[i]/0.15,
                          attack_rate=0.15,
                          detection_rate=0.8,
                          lethality=lethality_candidates[j],
                          infected_start=int(infected_start_candidates[k]),
                          contagious_start=int(infected_start_candidates[k]*2/3),
                          confirmed_start=confirmed_data[0])
world.change_n_p(days.size, 0.9/0.15)

for i in days_pred:
    world.update()
    infected[i], infected_confirmed[i] = world.infected_total, world.confirmed_total
    infected_day[i] = world.infected_day

fig, axs = plt.subplots(2, 1)
fig.set_figheight(10)
fig.set_figwidth(10)
for ax in axs:
    ax.set_xlabel("days")

axs[0].scatter(days, confirmed_data, marker='o', color='k', label='data (germany)')
axs[0].plot(days, infected_confirmed[:days.size], color='k', label='model')
axs[0].legend(loc="upper left")
axs[0].set_ylabel("Confirmed Cases")
axs[0].set_yscale("log")

axs[1].scatter(days, dead_data, marker='o', color='k', label='data (germany)')
axs[1].plot(days, np.cumsum(world.death_p_day)[:days.size], color='red', label='model')
axs[1].legend(loc="upper left")
axs[1].set_ylabel("Dead")
axs[1].set_yscale("log")
axs[0].legend(loc='upper left', fontsize=14)
axs[1].legend(loc='upper left', fontsize=14)
plt.savefig('img/fit_data_model.png', bbox_inches='tight')
plt.close("all")

# Prediction
fig, axs = plt.subplots(2, 2)
fig.set_figheight(9)
fig.set_figwidth(16)

axs[0, 0].plot(days_pred, infected, color='blue', label='total (model)')
axs[0, 0].plot(days_pred, infected_confirmed, color='k', label='confirmed (model)')
axs[0, 0].scatter(days, confirmed_data, marker='o', color='k', label='data (Germany)')
axs[0, 0].set_ylabel("Cases")
axs[0, 0].legend(loc='upper left', fontsize=14)

axs[1, 0].plot(days_pred, np.cumsum(world.death_p_day), color='r', label='model')
axs[1, 0].scatter(days, dead_data, marker='o', color='k', label='data (Germany)')
axs[1, 0].set_xlabel("Days")
axs[1, 0].set_ylabel("Deaths")
axs[1, 0].legend(loc='upper left', fontsize=14)

axs[0, 1].plot(days_pred, infected_day, color='b', ls="dashed", label='total (model)')
axs[0, 1].plot(days_pred, world.detect_p_day, color='k', ls="dashed", label='confirmed (model)')
axs[0, 1].scatter(days[1:], np.diff(confirmed_data), marker='o', color='k', label='data (Germany)')
axs[0, 1].set_ylabel("New cases per day")
axs[0, 1].legend(loc='upper left', fontsize=14)

axs[1, 1].plot(days_pred, world.death_p_day, color='r', ls="dashed", label='model')
axs[1, 1].scatter(days[1:], np.diff(dead_data), marker='o', color='k', label='data (Germany)')
axs[1, 1].legend(loc="upper left")
axs[1, 1].set_xlabel("Days")
axs[1, 1].set_ylabel("New deaths per day")
axs[1, 1].legend(loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig('img/fit_data_model_predict.png', bbox_inches='tight')
plt.close()
