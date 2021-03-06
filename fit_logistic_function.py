import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

import models
from load_data import load_jhu
from plotting import with_latex
mpl.rcParams.update(with_latex)

data = load_jhu()
os.makedirs('img', exist_ok=True)

confirmed = data.to_numpy()[36:, 0] - 16
days = np.arange(len(confirmed))

popt, pcov = curve_fit(models.logistic_function_growth, days, confirmed, p0=(1e5, 1e-6, 10), sigma=np.sqrt(confirmed),
                       absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))

plt.scatter(days, confirmed, marker='o', color='k', label='germany')
x = np.linspace(days[0], days[-1]*2.5, 300)

plt.plot(x, models.logistic_function_growth(x, *popt), color='red')
plt.plot(x, models.logistic_function_growth(x, popt[0]+perr[0], popt[1], popt[2]), color='red', ls='dotted')
plt.plot(x, models.logistic_function_growth(x, popt[0]-perr[0], popt[1], popt[2]), color='red', ls='dotted')
plt.axhline(popt[0], color='red', ls='dashed')
plt.text(days[int(np.rint(0.02*len(days)))], 0.9*popt[0], s='%i' % popt[0], color='red')
plt.axvline(45, color='C1', ls='dashed')
plt.text(46, 0.01*popt[0], s='eastern', color='C1')
plt.xlabel('days')
plt.ylabel('confirmed')
plt.savefig('img/fit_cases.png', bbox_inches='tight')
plt.close()

confirmed_day = np.diff(confirmed)
plt.scatter(days[1:], confirmed_day, marker='o', color='k', label='germany')
plt.xlabel('days')
plt.ylabel('confirmed / day')
plt.savefig('img/cases_day.png', bbox_inches='tight')
plt.close()
