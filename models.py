import numpy as np


def logistic_function(x, a, b):
    return a / (1 + np.exp(-b*a*x) * (a / 16))


class SimplePandemie:
    """ SimplePandemie """

    def __init__(self, n_p=15, attack_rate=0.15, t_contagious=4, t_cured=14, t_death=17, t_confirmed=6, infected_start=10,
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
        self.scale = 1      # scaling each case to save memory

        self.infected = np.zeros(nbuffer).astype(bool)
        self.infected[:infected_start] = True
        self.susceptible = total_population
        self.dead, self.cured, self.confirmed = 0, 0, 0
        self.days = np.zeros(nbuffer).astype(np.uint)
        self.days_to_contagious = np.zeros(nbuffer).astype(np.uint)
        self.days_to_death = -np.ones(nbuffer).astype(np.int)
        self.days_to_cure = -np.ones(nbuffer).astype(np.int)
        self.days_to_detect = -np.ones(nbuffer).astype(np.int)
        self.days_to_contagious[self.infected] = np.random.poisson(lam=t_contagious, size=infected_start)

    def infect(self):
        immune = self.cured + np.sum(self.infected)
        n_eff = self.n_p * (self.total_population - immune) / self.total_population
        n_infections = np.sum(np.random.poisson(n_eff*self.attack_rate, size=np.sum(self.is_contagious())))
        self.susceptible -= n_infections
        idx = np.where(~self.infected)[0][:n_infections]
        self.infected[idx] = True
        self.days_to_contagious[idx] = np.random.poisson(lam=self.t_contagious, size=n_infections)
        n_deaths = np.sum(np.random.rand(n_infections) <= self.lethality)
        idx_dying = idx[:n_deaths]
        idx_cure = idx[n_deaths:]
        self.days_to_death[idx_dying] = np.random.poisson(lam=self.t_death, size=len(idx_dying))
        self.days_to_cure[idx_cure] = np.random.poisson(lam=self.t_cured, size=len(idx_cure))
        n_confirmed = np.sum(np.random.rand(n_infections - n_deaths) <= self.detection_rate)
        idx_confirmed = idx[:(n_deaths + n_confirmed)]
        self.days_to_detect[idx_confirmed] = np.random.poisson(lam=self.t_confirmed, size=n_confirmed + n_deaths)

    def _reset(self, idx):
        self.infected[idx] = False
        self.days[idx] = 0
        self.days_to_contagious[idx] = 0
        self.days_to_death[idx] = -1
        self.days_to_cure[idx] = -1
        self.days_to_detect[idx] = -1

    def die(self):
        idx = self.is_dead()
        self.dead += np.sum(idx)
        self.total_population -= np.sum(idx)
        self._reset(idx)

    def cure(self):
        idx = self.is_cured()
        self.cured += np.sum(idx)
        self._reset(idx)

    def detect(self):
        idx = self.is_detected()
        self.confirmed += np.sum(idx)

    def is_contagious(self):
        return self.infected & (self.days == self.days_to_contagious)

    def is_dead(self):
        return self.infected & (self.days == self.days_to_death)

    def is_cured(self):
        return self.infected & (self.days == self.days_to_cure)

    def is_detected(self):
        return self.infected & (self.days == self.days_to_detect)

    def update(self):
        self.days[self.infected] += 1
        self.infect()
        self.detect()
        self.cure()
        self.die()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from latex_style import with_latex
    mpl.rcParams.update(with_latex)

    days = np.arange(100)
    infected = np.zeros(days.size)
    confirmed = np.zeros(days.size)
    cured = np.zeros(days.size)
    dead = np.zeros(days.size)
    world = SimplePandemie(lethality=0.2, detection_rate=1, total_population=10000, nbuffer=10000)
    for i in days:
        world.update()
        infected[i] = np.sum(world.infected)
        confirmed[i] = world.confirmed
        cured[i] = world.cured
        dead[i] = world.dead

    plt.plot(days, infected, color='blue', label='infected')
    plt.plot(days, confirmed, color='k', label='cases')
    plt.plot(days, dead, color='red', label='dead')
    plt.plot(days, cured, color='green', label='cured')
    plt.legend(loc='upper left')
    # plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("counts")
    plt.savefig('img/first_model.png', bbox_inches='tight')
    plt.close()
