import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import poisson, skewnorm


def logistic_function(x, a, b, c, d):
    return (a - b) / (1 + np.exp(d*(x - c))) + b


def logistic_function_growth(x, a, b, c):
    return a / (1 + (a / c - 1) * np.exp(-a*b*x))


def lognorm(x, mean=20.1, std=11.6, epsilon=1e-3):
    """ Log-normal distribution as used for time distribution until death """
    sigma = np.sqrt(np.log((std/mean)**2 + 1))
    mu = np.log(mean) - sigma**2 / 2
    x = x.astype(float) + epsilon
    norm = 1 / x / sigma / np.sqrt(2*np.pi)
    return norm * np.exp(-(np.log(x)-mu)**2 / 2 / sigma**2)


class DayDrivenPandemie(object):

    def __init__(self, n_days=100, n_p=15, attack_rate=0.15, t_contagious=4, t_cured=14, t_death=20, t_confirmed=6,
                 infected_start=10, lethality=0.01, detection_rate=0.8, total_population=83e6, contagious_start=7,
                 confirmed_start=7, death_pdf='skewnorm'):

        assert infected_start >= contagious_start, "More contagious than infected people!"
        assert infected_start >= confirmed_start, "More confirmed than infected people!"
        self.n_p = n_p
        self.attack_rate = attack_rate
        self.lethality = lethality
        self.detection_rate = detection_rate
        self.t_confirmed = tfp.distributions.Poisson(t_confirmed)
        self.t_death = tfp.distributions.Poisson(t_death)
        self.t_contagious = tfp.distributions.Poisson(t_contagious)
        self.t_cured = tfp.distributions.Poisson(t_cured)
        self.total_population = total_population
        # print('lethality: %s \tR0: %s \tdetection_rate: %s' % (lethality, attack_rate*n_p, detection_rate))

        self.n_p_steps = {}

        self.day = 0
        self.n_days = n_days
        self.infected_p_day = tf.Variable(initial_value=np.zeros(n_days), name='infected_p_day', dtype=tf.float32)
        self.infected_p_day = self.infected_p_day[0].assign(infected_start)
        self.confirmed_p_day = tf.Variable(initial_value=np.zeros(n_days), name='confirmed_p', dtype=tf.float32)
        self.confirmed_p_day = self.confirmed_p_day[0].assign(confirmed_start)
        self.death_p_day = tf.Variable(initial_value=np.zeros(n_days), name='death_p', dtype=tf.float32)
        self.contagious_p_day = tf.Variable(initial_value=np.zeros(n_days), name='contagious_p', dtype=tf.float32)
        self.contagious_p_day = self.contagious_p_day[0].assign(contagious_start)
        self.cured_p_day = tf.Variable(initial_value=np.zeros(n_days), name='cured_p', dtype=tf.float32)

        self._assign_timing(infected_start)

    def _assign_timing(self, n):
        n_death = n * self.lethality
        n_detected = self.detection_rate * (n - n_death) + n_death
        for d in range(self.n_days - self.day):
            self.confirmed_p_day = self.confirmed_p_day[self.day+d].assign(self.confirmed_p_day[self.day+d] + n_detected*self.t_confirmed.prob(d))
            self.death_p_day = self.death_p_day[self.day+d].assign(self.death_p_day[self.day+d] + n_death*self.t_death.prob(d))
            self.contagious_p_day = self.contagious_p_day[self.day+d].assign(self.contagious_p_day[self.day+d] + n*self.t_contagious.prob(d))
            self.cured_p_day = self.cured_p_day[self.day+d].assign(self.cured_p_day[self.day+d] + (n - n_death)*self.t_cured.prob(d))

    def infect(self):
        immune = tf.reduce_sum(self.infected_p_day[:(self.day+1)]) + tf.reduce_sum(self.cured_p_day[:(self.day+1)])
        n_eff = self.n_p * (self.total_population - immune) / self.total_population
        infected_day = self.contagious_p_day[self.day] * n_eff * self.attack_rate
        self.infected_p_day = self.infected_p_day[self.day].assign(self.infected_p_day[self.day] + infected_day)
        self._assign_timing(infected_day)

    def update(self, n_sim=1):
        for i in range(n_sim):
            if str(self.day) in self.n_p_steps:
                self.n_p = self.n_p_steps[str(self.day)]
            self.infect()
            self.infected_p_day = self.infected_p_day[self.day].assign(self.infected_p_day[self.day])
            self.day += 1

    def change_n_p(self, day, n_p):
        self.n_p_steps.update({str(day): n_p})


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from plotting import with_latex
    mpl.rcParams.update(with_latex)

    days = np.arange(100)
    world = DayDrivenPandemie(n_days=len(days), lethality=0.2, detection_rate=0.8)
    for i in days:
        world.update()

    plt.plot(days, tf.cumsum(world.infected_p_day).numpy(), color='blue', label='infected (total)')
    plt.plot(days, tf.cumsum(world.confirmed_p_day).numpy(), color='blue', ls='dashed', label='infected (confirmed)')
    plt.plot(days, world.infected_p_day.numpy(), color='k', label='new infections (total)')
    plt.plot(days, world.confirmed_p_day.numpy(), color='k', ls='dashed', label='new infections (confirmed)')
    plt.plot(days, tf.cumsum(world.death_p_day).numpy(), color='red', label='dead')
    plt.plot(days, tf.cumsum(world.cured_p_day).numpy(), color='green', label='cured')
    plt.legend(loc='upper left', fontsize=14)
    # plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("counts")
    plt.savefig('img/first_model_tf_day.png', bbox_inches='tight')
    plt.close()
