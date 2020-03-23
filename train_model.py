import numpy as np
from scipy.stats import poisson
from models import DayDrivenPandemie


def likelihood(cases, cases_expect, deaths, deaths_expect):
    # get probability mass function and catch floating precision
    likelihood_cases = np.log10(np.clip(poisson.pmf(cases_expect, mu=cases), a_min=1e-11, a_max=None))
    likelihood_deaths = np.log10(np.clip(poisson.pmf(deaths_expect, mu=deaths), a_min=1e-11, a_max=None))
    return np.sum(likelihood_cases) + np.sum(likelihood_deaths)


def run_model(pars, n_sim, n_burn_in=5, day_action=None):
    # day_action=18 corresponds to Monday (March 16, 2019)
    np.random.seed(0)
    if day_action is None:
        n_burn_in = pars[1]     # run simulation n_burn_in days before data taking

    cases, confirmed, dead = np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim)
    world = DayDrivenPandemie(n_days=n_sim,
                              n_p=pars[2]/0.15,
                              attack_rate=0.15,
                              detection_rate=0.8,
                              lethality=pars[0],
                              infected_start=50,
                              contagious_start=0,
                              confirmed_start=0)

    if day_action is not None:
        r0_action = pars[1]     # change R0 at day_action
        world.change_n_p(n_burn_in + day_action, r0_action)

    for i in np.arange(n_sim):
        world.update()
        cases[i], confirmed[i], dead[i] = world.infected_total, world.confirmed_total, world.dead

    return cases, confirmed, dead


def sample_likelihood(pars, confirmed_day_data, dead_day_data, day_action=None):

    n_sim = len(confirmed_day_data) + 1
    if day_action is None:
        n_sim += pars[1]
    _, confirmed, dead = run_model(pars, n_sim, day_action)
    confirmed_day = np.diff(confirmed)[-len(confirmed_day_data):]
    dead_day = np.diff(dead)[-len(confirmed_day_data):]

    return -likelihood(confirmed_day, confirmed_day_data, dead_day, dead_day_data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.optimize import minimize
    from latex_style import with_latex
    from plotting import load_data
    mpl.rcParams.update(with_latex)

    data = load_data()
    confirmed_data, dead_data = data.to_numpy()[36:, 0] - 16, data.to_numpy()[36:, 1]
    confirmed_day_data, dead_day_data = np.diff(confirmed_data), np.diff(dead_data)
    days = np.arange(len(confirmed_data))

    popt = minimize(sample_likelihood, x0=[0.03, 2.7, 1.], method='L-BFGS-B',
                    bounds=[(0.005, 0.1), (2., 3.5), (0.5, 3.)],
                    args=(confirmed_day_data, dead_day_data, 18))
    print(popt)
