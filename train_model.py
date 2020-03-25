import numpy as np
from scipy.stats import poisson
from models import DayDrivenPandemie


def likelihood(cases, cases_expect, deaths, deaths_expect):

    # get probability mass function and catch floating precision
    likelihood_cases = np.log10(np.clip(poisson.pmf(cases_expect, mu=cases), a_min=1e-11, a_max=None))
    likelihood_deaths = np.log10(np.clip(poisson.pmf(deaths_expect, mu=deaths), a_min=1e-11, a_max=None))
    _likelihood = np.sum(likelihood_cases) + np.sum(likelihood_deaths)

    likelihood_cases_0 = np.log10(np.clip(poisson.pmf(cases_expect, mu=cases_expect), a_min=1e-11, a_max=None))
    likelihood_deaths_0 = np.log10(np.clip(poisson.pmf(deaths_expect, mu=deaths_expect), a_min=1e-11, a_max=None))
    _likelihood_0 = np.sum(likelihood_cases_0) + np.sum(likelihood_deaths_0)

    return 2 * (_likelihood - _likelihood_0)


def run_model(pars, n_sim, n_burn_in=9, day_action=None):

    day_action = pars['day-action'] if ('day-action' in pars) else day_action
    cases, confirmed, dead, active = np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim)
    world = DayDrivenPandemie(n_days=n_sim,
                              n_p=pars.get('R0-0', 2.7)/0.15,
                              attack_rate=pars.get('attack-rate', 0.15),
                              detection_rate=pars.get('detection-rate', 0.6),
                              lethality=pars.get('lethality', 0.01),
                              t_contagious=pars.get('t-contagious', 4),
                              t_cured=pars.get('t-cured', 14),
                              t_death=pars.get('t-death', 12),
                              t_confirmed=pars.get('t-confirmed', 6),
                              infected_start=pars.get('infected-start', 30),
                              contagious_start=0,
                              confirmed_start=0)

    if ('R0-1' in pars):
        n_burn_in = n_burn_in if ('burn-in' not in pars) else pars['burn-in']
        world.change_n_p(n_burn_in + day_action, pars.get('R0-1')/0.15)  # change R0 at day_action
    if ('R0-2' in pars):
        n_burn_in = n_burn_in if ('burn-in' not in pars) else pars['burn-in']
        world.change_n_p(n_burn_in + 24, pars.get('R0-2')/0.15)  # change R0 at day_action

    for i in np.arange(n_sim):
        world.update()
        cases[i], confirmed[i], dead[i] = world.infected_total, world.confirmed_total, world.dead
        active[i] = world.infected_total - world.dead - world.cured

    return cases, confirmed, dead, active


def sample_likelihood(pars, confirmed_day_data, dead_day_data, day_action=None):

    n_burn_in = pars['burn-in'] if ('burn-in' in pars) else 9
    day_action = pars['day-action'] if ('day-action' in pars) else day_action
    n_sim = len(confirmed_day_data) + n_burn_in + 1
    _, confirmed, dead, _ = run_model(pars, n_sim, n_burn_in, day_action)
    confirmed_day = np.diff(confirmed)[-len(confirmed_day_data):]
    dead_day = np.diff(dead)[-len(confirmed_day_data):]

    return -likelihood(confirmed_day, confirmed_day_data, dead_day, dead_day_data)
