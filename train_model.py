import numpy as np
from scipy.stats import poisson


def likelihood(cases, cases_expect, deaths, deaths_expect):
    # get probability mass function and catch floating precision
    likelihood_cases = np.log10(np.clip(poisson.pmf(cases_expect, mu=cases), a_min=1e-11, a_max=None))
    likelihood_deaths = np.log10(np.clip(poisson.pmf(deaths_expect, mu=deaths), a_min=1e-11, a_max=None))
    return np.sum(likelihood_cases) + np.sum(likelihood_deaths)
