import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


class world(object):
    def __init__(self, n_days, t_contagious):
        self.n_days = n_days
        self.tdist = tfp.distributions.Poisson(rate=t_contagious)
        self.days = tf.range(0, self.n_days)

    def _reset(self):
        self.contagious_p_day = tf.concat(
            [tf.ones(1) * self.i_start, tf.zeros(self.n_days - 1)], axis=0
        )

    def graph(self, log_R0, log_i_start):
        self.R0 = tf.math.exp(log_R0)
        self.i_start = tf.math.exp(log_i_start)
        self._reset()
        for day in self.days:
            drange = tf.range(
                0, limit=tf.cast(self.n_days - day, tf.float32), dtype=tf.float32
            )
            infect_new = self.contagious_p_day[day] * self.R0
            self.contagious_p_day = self.contagious_p_day + tf.concat(
                [tf.zeros(day), infect_new * self.tdist.prob(drange)], axis=0
            )

    def loss(self, data):
        return tf.losses.MSE(data, self.contagious_p_day)


def main():
    n_days = 20
    n_steps = int(1e2)
    t_contagious = 3
    data = tf.math.exp(0.3 * tf.range(0, n_days, dtype=tf.float32))
    log_R0 = tf.Variable(0.5, dtype=tf.float32, name="log_R0")
    log_i_start = tf.Variable(1.0, dtype=tf.float32, name="log_infected_start")
    variables = [log_R0, log_i_start]

    model = world(n_days, t_contagious)
    optimizer = tf.optimizers.SGD(0.0001)

    def train_step():
        with tf.GradientTape() as tape:
            model.graph(*variables)
            loss = model.loss(data)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    for i in range(n_steps):
        loss = train_step()
        print("loss: {}, R0: {}, i_start: {}".format(loss, model.R0, model.i_start))

    fig, axs = plt.subplots(1, 1)
    axs.scatter(
        model.days.numpy(),
        model.contagious_p_day.numpy(),
        c="b",
        label="model",
        alpha=0.5,
    )
    axs.scatter(model.days.numpy(), data.numpy(), c="r", label="data", alpha=0.5)
    axs.legend()
    axs.set_title(
        "Optimal: R0={}, i_start={}, final loss={}".format(
            model.R0.numpy(), model.i_start.numpy(), loss.numpy()
        )
    )
    plt.show()


if __name__ == "__main__":
    main()
