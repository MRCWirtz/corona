import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import models
from latex_style import with_latex
mpl.rcParams.update(with_latex)

cases = np.array([16, 18, 21, 26, 53, 66, 117, 150, 188, 240, 400, 639, 795, 902, 1139, 1296, 1567, 2369, 3062, 3795,
                  4838, 6012])
days = np.arange(len(cases))
os.makedirs('img', exist_ok=True)

popt, pcov = curve_fit(models.logistic_function, days, cases, p0=(50000, 1e-3), sigma=np.sqrt(cases), absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
plt.scatter(days, cases, marker='o', color='k', label='germany')
x = np.linspace(days[0], days[-1]*2.5, 300)
plt.plot(x, models.logistic_function(x, *popt), color='red')
plt.plot(x, models.logistic_function(x, popt[0]+perr[0], popt[1]-perr[1]), color='red', ls='dashed')
plt.plot(x, models.logistic_function(x, popt[0]-perr[0], popt[1]+perr[1]), color='red', ls='dashed')
plt.axhline(popt[0], color='red', ls='dotted')
plt.text(days[int(np.rint(0.02*len(days)))], 0.92*popt[0], s='%i cases' % popt[0], color='red')
plt.axvline(45, color='C1', ls='dashed')
plt.text(46, 0.01*popt[0], s='eastern', color='C1')
plt.xlabel('days')
plt.ylabel('cases')
plt.savefig('img/fit_cases.png', bbox_inches='tight')
plt.close()

cases_day = np.diff(cases)
plt.scatter(days[1:], cases_day, marker='o', color='k', label='germany')
plt.xlabel('days')
plt.ylabel('cases / day')
plt.savefig('img/cases_day.png', bbox_inches='tight')
plt.close()
