import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class DayDrivenPandemie(object):

    def __init__(self, n_days=100, n_p=15, attack_rate=0.15, t_contagious=4, t_cured=14, t_death=20, t_confirmed=6,
                 infected_start=10, lethality=0.01, detection_rate=0.8, total_population=83e6, confirmed_start=0,
                 death_pdf='skewnorm'):

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
        self.cured_p_day = tf.Variable(initial_value=np.zeros(n_days), name='cured_p', dtype=tf.float32)

        self._assign_timing(infected_start)

    def _assign_timing(self, n):
        n_death = n * self.lethality
        n_detected = self.detection_rate * (n - n_death) + n_death
        days = tf.range(start=0, limit=self.n_days-self.day, dtype=tf.float32)
        self.confirmed_p_day[self.day:].assign(self.confirmed_p_day[self.day:] + n_detected*self.t_confirmed.prob(days))
        self.death_p_day[self.day:].assign(self.death_p_day[self.day:] + n_death*self.t_death.prob(days))
        self.contagious_p_day[self.day:].assign(self.contagious_p_day[self.day:] + n*self.t_contagious.prob(days))
        self.cured_p_day[self.day:].assign(self.cured_p_day[self.day:] + (n - n_death)*self.t_cured.prob(days))

    def infect(self):
        immune = tf.reduce_sum(self.infected_p_day[:(self.day+1)])
        n_eff = self.n_p * (self.total_population - immune) / self.total_population
        infected_day = self.contagious_p_day[self.day] * n_eff * self.attack_rate
        self.infected_p_day[self.day].assign(self.infected_p_day[self.day] + infected_day)
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
    plt.plot(days, tf.cumsum(world.confirmed_p_day).numpy(), color='blue', ls='dashed', label='confirmed (total)')
    plt.plot(days, world.infected_p_day.numpy(), color='k', label='new infections')
    plt.plot(days, world.confirmed_p_day.numpy(), color='k', ls='dashed', label='new confirmed')
    plt.plot(days, tf.cumsum(world.death_p_day).numpy(), color='red', label='dead')
    plt.plot(days, tf.cumsum(world.cured_p_day).numpy(), color='green', label='cured')
    plt.legend(loc='upper left', fontsize=14)
    # plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("counts")
    plt.savefig('img/first_model_tf_day.png', bbox_inches='tight')
    plt.close()
