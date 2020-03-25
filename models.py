import numpy as np
from scipy.stats import poisson, skewnorm


def logistic_function(x, a, b, c):
    return a / (1 + (a / c - 1) * np.exp(-a*b*x))


def lognorm(x, mean=20.1, std=11.6):
    """ Log-normal distribution as used for time distribution until death """
    sigma = np.sqrt(np.log((std/mean)**2 + 1))
    mu = np.log(mean) - sigma**2 / 2
    norm = 1 / x / sigma / np.sqrt(2*np.pi)
    return norm * np.exp(-(np.log(x)-mu)**2 / 2 / sigma**2)


class IndividuumDrivenPandemie:
    """ SimplePandemie """

    def __init__(self, n_p=15, attack_rate=0.15, t_contagious=4, t_cured=14, t_death=12, t_confirmed=6, infected_start=10,
                 lethality=0.01, detection_rate=0.8, nbuffer=10000, total_population=83e6):
        self.n_p = n_p
        self.attack_rate = attack_rate
        self.lethality = lethality
        self.detection_rate = detection_rate
        self.t_contagious = t_contagious
        self.t_cured = t_cured
        self.t_death = t_death
        self.t_confirmed = t_confirmed
        self.total_population = total_population

        self.infected = np.zeros(nbuffer).astype(bool)
        self.infected[:infected_start] = True
        self.susceptible = total_population
        self.infected_total, self.infected_total_confirmed = infected_start, 0
        self.infected_day, self.infected_day_confirmed = 0, 0
        self.dead, self.dead_day, self.cured = 0, 0, 0
        self.days = np.zeros(nbuffer).astype(np.uint)
        self.days_to_contagious = np.zeros(nbuffer).astype(np.uint)
        self.days_to_death = -np.ones(nbuffer).astype(np.int)
        self.days_to_cure = -np.ones(nbuffer).astype(np.int)
        self.days_to_detect = -np.ones(nbuffer).astype(np.int)
        self._assign_timing(np.arange(infected_start))

        self.scale = 1
        self.fraction_buffer = 0
        self.scale_each = np.ones(nbuffer).astype(int)

    def infect(self):
        immune = self.cured + np.sum(self.scale_each*self.infected)
        n_eff = self.n_p * (self.total_population - immune) / self.total_population
        self.infected_day = np.sum(np.random.poisson(n_eff*self.attack_rate, size=np.sum(self.scale_each*self.is_contagious())))
        self.infected_total += self.infected_day
        self.susceptible -= self.infected_day
        selection = np.cumsum(~self.infected * self.scale_each) <= self.infected_day
        select_idx = np.where(~self.infected * selection)[0]
        self.infected[select_idx] = True
        self.fraction_buffer = len(select_idx) / len(self.infected)
        self._assign_timing(select_idx)

    def detect(self):
        self.infected_day_confirmed = np.sum(self.scale_each * self.is_detected())
        self.infected_total_confirmed += self.infected_day_confirmed

    def cure(self):
        mask_cured = self.is_cured()
        self.cured += np.sum(self.scale_each*mask_cured)
        self._reset(mask_cured)

    def die(self):
        mask_die = self.is_dead()
        self.dead_day = np.sum(self.scale_each*mask_die)
        self.dead += self.dead_day
        self.total_population -= self.dead_day
        self._reset(mask_die)

    def _assign_timing(self, idx):
        self.days_to_contagious[idx] = np.random.poisson(lam=self.t_contagious, size=len(idx))
        n_deaths = np.random.binomial(len(idx), self.lethality)
        idx_dying = np.random.choice(idx, size=n_deaths, replace=False)
        idx_cure = idx[np.in1d(idx, idx_dying, invert=True)]
        self.days_to_death[idx_dying] = np.random.poisson(lam=self.t_death, size=len(idx_dying))
        self.days_to_cure[idx_cure] = np.random.poisson(lam=self.t_cured, size=len(idx_cure))
        n_confirmed = np.random.binomial(len(idx_cure), self.detection_rate)
        idx_confirmed = np.append(idx_dying, np.random.choice(idx_cure, size=n_confirmed, replace=False))
        self.days_to_detect[idx_confirmed] = np.random.poisson(lam=self.t_confirmed, size=len(idx_confirmed))

    def _reset(self, mask):
        self.infected[mask] = False
        self.days[mask] = 0
        self.days_to_contagious[mask] = 0
        self.days_to_death[mask] = -1
        self.days_to_cure[mask] = -1
        self.days_to_detect[mask] = -1
        self.scale_each[mask] = self.scale

    def _scale(self):
        if (self.fraction_buffer > 0.001) or (np.sum(self.infected) / len(self.infected) > 0.5):
            self.scale += max(1, int(self.fraction_buffer * 100000))
        else:
            self.scale = max(1, self.scale-1)

    def is_contagious(self):
        return self.infected & (self.days == self.days_to_contagious)

    def is_dead(self):
        return self.infected & (self.days == self.days_to_death)

    def is_cured(self):
        return self.infected & (self.days == self.days_to_cure)

    def is_detected(self):
        return self.infected & (self.days == self.days_to_detect)

    def update(self):
        self.infect()
        self.detect()
        self.cure()
        self.die()
        # self._scale()
        # print('slots used: %s \tscale=%s' % (np.sum(self.infected), self.scale))
        self.days[self.infected] += 1


class DayDrivenPandemie(object):

    def __init__(self, n_days=100, n_p=15, attack_rate=0.15, t_contagious=4, t_cured=14, t_death=20, t_confirmed=6,
                 infected_start=10, lethality=0.01, detection_rate=0.8, total_population=83e6, contagious_start=7,
                 confirmed_start=7):

        assert infected_start >= contagious_start, "More contagious than infected people!"
        assert infected_start >= confirmed_start, "More confirmed than infected people!"
        self.n_p = n_p
        self.attack_rate = attack_rate
        self.lethality = lethality
        self.detection_rate = detection_rate
        self.t_contagious = t_contagious
        self.t_cured = t_cured
        self.t_death = t_death
        self.t_confirmed = t_confirmed
        self.total_population = total_population

        self.n_p_steps = {}

        self.day = 0
        self.n_days = n_days
        self.contagious_p_day = np.zeros(n_days)
        self.death_p_day = np.zeros(n_days)
        self.cured_p_day = np.zeros(n_days)
        self.detect_p_day = np.zeros(n_days)

        self.infected, self.contagious = infected_start, contagious_start
        self.infected_total, self.confirmed_total = infected_start, confirmed_start
        self.infected_day = 0
        self.dead, self.dead_day, self.cured = 0, 0, 0
        self._assign_timing(infected_start)

    def _count_p_days(self, n, t, pdf='poisson'):
        if pdf == 'poisson':
            p_days = n * poisson.pmf(np.arange(self.n_days - self.day), mu=t)
        elif pdf == 'skewnorm':
            p_days = n * skewnorm.pdf(np.arange(self.n_days - self.day), a=5, loc=t, scale=15)
        elif pdf == 'lognorm-poisson':
            _t = np.arange(self.n_days - self.day)
            p_days = n * np.convolve(poisson.pmf(_t, mu=t), lognorm(_t), mode='full')[:len(_t)]
        else:
            raise NotImplementedError("Density function pdf='%s' not implemented!" % pdf)
        return np.pad(p_days, (self.day, 0), mode='constant')

    def _assign_timing(self, n):
        n_death = n * self.lethality
        n_detected = self.detection_rate * (n - n_death) + n_death
        self.contagious_p_day += self._count_p_days(n, self.t_contagious)
        self.cured_p_day += self._count_p_days(n - n_death, self.t_cured)
        self.death_p_day += self._count_p_days(n_death, self.t_death, pdf='skewnorm')
        self.detect_p_day += self._count_p_days(n_detected, self.t_confirmed)

    def infect(self):
        immune = self.infected + self.cured
        n_eff = self.n_p * (self.total_population - immune) / self.total_population
        self.infected_day = self.contagious_p_day[self.day] * n_eff*self.attack_rate
        # self.infected_day = np.sum(np.random.poisson(n_eff*self.attack_rate, size=self.contagious_p_day[self.day]))
        self.infected += self.infected_day
        self.infected_total += self.infected_day
        self._assign_timing(self.infected_day)

    def update(self):
        if str(self.day) in self.n_p_steps:
            self.n_p = self.n_p_steps[str(self.day)]
        self.infect()
        self.infected -= (self.cured_p_day[self.day] + self.death_p_day[self.day])
        self.contagious += self.contagious_p_day[self.day] - self.cured_p_day[self.day] - self.death_p_day[self.day]
        self.cured += self.cured_p_day[self.day]
        self.dead += self.death_p_day[self.day]
        self.confirmed_total += self.detect_p_day[self.day]
        self.day += 1

    def change_n_p(self, day, n_p):
        self.n_p_steps.update({str(day): n_p})


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from latex_style import with_latex
    mpl.rcParams.update(with_latex)

    days = np.arange(42)
    infected, infected_confirmed = np.zeros(days.size), np.zeros(days.size)
    infected_day, infected_day_confirmed = np.zeros(days.size), np.zeros(days.size)
    cured, dead = np.zeros(days.size), np.zeros(days.size)
    world = DayDrivenPandemie(lethality=0.2, detection_rate=0.8)
    for i in days:
        world.update()
        infected[i], infected_confirmed[i] = world.infected_total, world.infected_total_confirmed
        infected_day[i], infected_day_confirmed[i] = world.infected_day, world.infected_day_confirmed
        cured[i], dead[i] = world.cured, world.dead

    plt.plot(days, infected, color='blue', label='infected (total)')
    plt.plot(days, infected_confirmed, color='blue', ls='dashed', label='infected (confirmed)')
    plt.plot(days, infected_day, color='k', label='new infections (total)')
    plt.plot(days, infected_day_confirmed, color='k', ls='dashed', label='new infections (confirmed)')
    plt.plot(days, dead, color='red', label='dead')
    plt.plot(days, cured, color='green', label='cured')
    plt.legend(loc='upper left', fontsize=14)
    plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("counts")
    plt.savefig('img/first_model_day.png', bbox_inches='tight')
    plt.close()
