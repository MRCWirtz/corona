import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

from models import DayDrivenPandemie
from load_data import load_data
from latex_style import with_latex
mpl.rcParams.update(with_latex)

np.random.seed(1)

data = load_data()
os.makedirs('img', exist_ok=True)

confirmed_data, dead_data = data.to_numpy()[36:, 0] - 16, data.to_numpy()[36:, 1]
days = np.arange(len(confirmed_data))
# x = np.linspace(days[0], days[-1]*2.5, 300)
#
# plt.plot(x, models.logistic_function(x, *popt), color='red')
# plt.plot(x, models.logistic_function(x, popt[0]+perr[0], popt[1], popt[2]), color='red', ls='dotted')
# plt.plot(x, models.logistic_function(x, popt[0]-perr[0], popt[1], popt[2]), color='red', ls='dotted')
# plt.axhline(popt[0], color='red', ls='dashed')
# plt.text(days[int(np.rint(0.02*len(days)))], 0.9*popt[0], s='%i' % popt[0], color='red')
# plt.axvline(45, color='C1', ls='dashed')
# plt.text(46, 0.01*popt[0], s='eastern', color='C1')
# plt.xlabel('days')
# plt.ylabel('confirmed')
# plt.savefig('img/fit_cases.png', bbox_inches='tight')
# plt.close()
#
# confirmed_day = np.diff(confirmed)
# plt.scatter(days[1:], confirmed_day, marker='o', color='k', label='germany')
# plt.xlabel('days')
# plt.ylabel('confirmed / day')
# plt.savefig('img/cases_day.png', bbox_inches='tight')
# plt.close()
#
# mpl.rcParams.update(with_latex)

pred_len = 60
days_pred = np.arange(days.size + pred_len)
infected, infected_confirmed = np.zeros(days_pred.size), np.zeros(days_pred.size)
infected_day, infected_day_confirmed = np.zeros(days_pred.size), np.zeros(days_pred.size)
cured, dead = np.zeros(days_pred.size), np.zeros(days_pred.size)
world = DayDrivenPandemie(n_days=days.size+pred_len,
                          n_p=17,
                          attack_rate=0.15,
                          detection_rate=0.8,
                          lethality=0.03,
                          infected_start=250,
                          contagious_start=200,
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
fig, axs = plt.subplots(2, 1)
fig.set_figheight(10)
fig.set_figwidth(10)

axs[0].plot(days_pred, infected, color='blue', label='total infected (model)')
axs[0].scatter(days, confirmed_data, marker='o', color='k', label='data (germany)')
axs[0].plot(days_pred, infected_confirmed, color='k', label='confirmed (model)')
axs[0].plot(days_pred, infected_day, color='b', ls="dashed", label='daily infections (model)')
axs[0].legend(loc="upper left")
axs[0].set_ylabel("Confirmed Cases")
# axs[0].set_yscale("log")
axs[0].legend(loc='upper left', fontsize=14)

axs[1].plot(days_pred, np.cumsum(world.death_p_day), color='red', label='model')
axs[1].scatter(days, dead_data, marker='o', color='k', label='data (germany)')
axs[1].legend(loc="upper left")
axs[1].set_ylabel("Dead")
# axs[1].set_yscale("log")
axs[1].legend(loc='upper left', fontsize=14)
plt.savefig('img/fit_data_model_predict.png', bbox_inches='tight')
plt.close()
