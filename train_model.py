import numpy as np
import scipy as sp
from scipy.stats import poisson


def likelihood(cases, cases_expect, deaths, deaths_expect):
    likelihood_cases = np.log10(poisson.pmf(cases_expect, mu=cases))
    likelihood_deaths = np.log10(poisson.pmf(deaths_expect, mu=deaths))
    return np.sum(likelihood_cases) + np.sum(likelihood_deaths)
