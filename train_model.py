import numpy as np
from scipy.stats import poisson
from models import logistic_function, DayDrivenPandemie
from parameters import defaults


def likelihood(cases, cases_expect, deaths, deaths_expect, eps=1e-11):

    # get probability mass function and catch floating precision
    likelihood_cases = np.log10(np.clip(poisson.pmf(cases_expect, mu=cases), a_min=eps, a_max=None))
    likelihood_deaths = np.log10(np.clip(poisson.pmf(deaths_expect, mu=deaths), a_min=eps, a_max=None))
    _likelihood = 0.2 * np.sum(likelihood_cases) + np.sum(likelihood_deaths)

    likelihood_cases_0 = np.log10(np.clip(poisson.pmf(cases_expect, mu=cases_expect), a_min=eps, a_max=None))
    likelihood_deaths_0 = np.log10(np.clip(poisson.pmf(deaths_expect, mu=deaths_expect), a_min=eps, a_max=None))
    _likelihood_0 = 0.2 * np.sum(likelihood_cases_0) + np.sum(likelihood_deaths_0)

    return 2 * (_likelihood - _likelihood_0)


def chi2_loss(cases, cases_expect, deaths, deaths_expect, eps=1e-11):

    # get probability mass function and catch floating precision
    mask_cases = cases_expect >= 1
    chi2_cases = np.mean((np.log10(cases[mask_cases]) - np.log10(cases_expect[mask_cases]))**2)
    mask_deaths = deaths_expect >= 1
    chi2_deaths = np.mean((np.log10(deaths[mask_deaths]) - np.log10(deaths_expect[mask_deaths]))**2)
    return chi2_cases + chi2_deaths


def likelihood_cum(cases, cases_expect, deaths, deaths_expect, eps=1e-11):

    cases_day, cases_exp_day = np.diff(cases), np.diff(cases_expect)
    deaths_day, deaths_exp_day = np.diff(deaths), np.diff(deaths_expect)
    # get probability mass function and catch floating precision
    likelihood_cases = np.log10(np.clip(poisson.pmf(cases_expect[1:], mu=cases_day, loc=cases_expect[:-1]), a_min=eps, a_max=None))
    likelihood_deaths = np.log10(np.clip(poisson.pmf(deaths_expect[1:], mu=deaths_day, loc=deaths_expect[:-1]), a_min=eps, a_max=None))
    _likelihood = np.sum(likelihood_cases) + np.sum(likelihood_deaths)

    likelihood_cases_0 = np.log10(np.clip(poisson.pmf(cases_expect[1:], mu=cases_exp_day, loc=cases_expect[:-1]), a_min=eps, a_max=None))
    likelihood_deaths_0 = np.log10(np.clip(poisson.pmf(deaths_expect[1:], mu=deaths_exp_day, loc=deaths_expect[:-1]), a_min=eps, a_max=None))
    _likelihood_0 = np.sum(likelihood_cases_0) + np.sum(likelihood_deaths_0)

    return 2 * (_likelihood - _likelihood_0)


def run_model(pars, n_sim):

    n_burn_in = pars['burn-in'] if ('burn-in' in pars) else defaults['burn-in']
    R0 = pars.get('R0-0', defaults['R0-0'])
    world = DayDrivenPandemie(n_days=n_sim+n_burn_in,
                              n_p=R0/defaults['attack-rate'],
                              attack_rate=pars.get('attack-rate', defaults['attack-rate']),
                              detection_rate=pars.get('detection-rate', defaults['detection-rate']),
                              lethality=pars.get('lethality', defaults['lethality']),
                              t_contagious=pars.get('t-contagious', defaults['t-contagious']),
                              t_cured=pars.get('t-cured', defaults['t-cured']),
                              t_death=pars.get('t-death', defaults['t-death']),
                              t_confirmed=pars.get('t-confirmed', defaults['t-confirmed']),
                              infected_start=pars.get('infected-start', defaults['infected-start']),
                              contagious_start=pars.get('contagious-start', defaults['contagious-start']),
                              confirmed_start=pars.get('confirmed-start', defaults['confirmed-start']))

    if ('R0-1' in pars):
        world.change_n_p(n_burn_in + defaults['day-action-1'], pars.get('R0-1')/defaults['attack-rate'])
    if ('R0-2' in pars):
        n_burn_in = pars['burn-in'] if ('burn-in' in pars) else defaults['burn-in']
        world.change_n_p(n_burn_in + defaults['day-action-2'], pars.get('R0-2')/defaults['attack-rate'])

    # Run burn-in phase without writing output
    world.update(n_sim=n_burn_in)

    # Run the model
    cases, confirmed, dead, active = np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim)
    for i in np.arange(n_sim):
        if ('R0-lo-A' in pars) or ('R0-lo-B' in pars) or ('R0-lo-C' in pars):
            _R0 = logistic_function(i, R0, pars.get('R0-lo-A'), pars.get('R0-lo-B'), pars.get('R0-lo-C'))
            world.change_n_p(n_burn_in + i, _R0/defaults['attack-rate'])
        world.update()
        cases[i], confirmed[i], dead[i] = world.infected_total, world.confirmed_total, world.dead
        active[i] = world.infected_total - world.dead - world.cured

    return cases, confirmed, dead, active


def sample_likelihood(pars, confirmed_data, dead_data, loss='chi2'):

    n_sim = max(len(confirmed_data), len(dead_data))
    _, confirmed, dead, _ = run_model(pars, n_sim)
    # If 'confirmed_data' and 'dead_dat' have different sizes, the last days will be discarded
    confirmed, dead = confirmed[:len(confirmed_data)], dead[:len(dead_data)]

    if loss == 'chi2':
        return chi2_loss(confirmed, confirmed_data, dead, dead_data)
    else:
        return -likelihood(np.diff(confirmed), np.diff(confirmed_data), np.diff(dead), np.diff(dead_data))
