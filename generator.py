import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose


class Conv2dLSTMCell(tf.keras.Model):
    def __init__(self, latent_dim, kernel_size=5):
        super().__init__()

        args = [latent_dim, kernel_size]
        kwargs = {'padding': 'SAME'}

        self.forget = Conv2D(*args, **kwargs, activation=tf.sigmoid)
        self.inp = Conv2D(*args, **kwargs, activation=tf.sigmoid)
        self.outp = Conv2D(*args, **kwargs, activation=tf.sigmoid)
        self.state = Conv2D(*args, **kwargs, activation=tf.tanh)

    def call(self, input, cell):
        forget_gate = self.forget(input)
        input_gate = self.inp(input)
        output_gate = self.outp(input)
        state_gate = self.state(input)

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * tf.tanh(cell)

        return hidden, cell


class LatentDistribution(tf.keras.Model):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.parametrize = Conv2D(z_dim * 2, 5, padding='SAME')

    def call(self, input):
        parametrization = self.parametrize(input)
        mu, sigma = tf.split(parametrization, [self.z_dim, self.z_dim], -1)
        return tf.distributions.Normal(loc=mu, scale=tf.nn.softplus(sigma))


class Generator(tf.keras.Model):
    def __init__(self, x_dim, z_dim, h_dim, L):
        super().__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.inference_core = Conv2dLSTMCell(h_dim)
        self.generator_core = Conv2dLSTMCell(h_dim)

        self.posterior_distribution = LatentDistribution(z_dim)
        self.prior_distribution = LatentDistribution(z_dim)

        self.observation_density = Conv2D(
            x_dim, 1, padding='SAME', activation=tf.sigmoid)

        self.upsample = Conv2DTranspose(h_dim, 4, strides=4)
        self.downsample = Conv2D(h_dim, 4, strides=4)

    def call(self, x, v, r):
        batch_size, v_dim = v.shape
        batch_size, im_size, im_size, x_dim = x.shape
        batch_size, r_size, r_size, r_dim = r.shape

        v = tf.tile(v, [1, r_size * r_size])
        v = tf.reshape(v, [-1, r_size, r_size, v_dim])

        kl = 0
        _, im_size, im_size, _ = x.shape

        c_g = tf.zeros([batch_size, r_size, r_size, self.h_dim])
        h_g = tf.zeros([batch_size, r_size, r_size, self.h_dim])
        u = tf.zeros([batch_size, im_size, im_size, self.h_dim])
        c_i = tf.zeros([batch_size, r_size, r_size, self.h_dim])
        h_i = tf.zeros([batch_size, r_size, r_size, self.h_dim])

        x = self.downsample(x)

        for _ in range(self.L):
            prior_factor = self.prior_distribution(h_g)
            input = tf.concat([h_i, h_g, x, v, r], 3)
            h_i, c_i = self.inference_core(input, c_i)

            posterior_factor = self.posterior_distribution(h_i)
            z = posterior_factor.sample()

            input = tf.concat([h_g, z, v, r], 3)
            h_g, c_g = self.generator_core(input, c_g)
            u = self.upsample(h_g) + u

            kl += tf.reduce_mean(
                tf.distributions.kl_divergence(posterior_factor, prior_factor)
            )

        x_mu = self.observation_density(u)

        return x_mu, kl

    def sample(self, v, r, im_size):
        batch_size, v_dim = v.shape
        batch_size, r_size, r_size, r_dim = r.shape

        v = tf.tile(v, [1, r_size * r_size])
        v = tf.reshape(v, [-1, r_size, r_size, v_dim])

        c_g = tf.zeros([batch_size, r_size, r_size, self.h_dim])
        h_g = tf.zeros([batch_size, r_size, r_size, self.h_dim])
        u = tf.zeros([batch_size, im_size, im_size, self.h_dim])

        for _ in range(self.L):
            prior_factor = self.prior_distribution(h_g)
            z = prior_factor.sample()
            input = tf.concat([h_g, z, v, r], 3)
            h_g, c_g = self.generator_core(input, c_g)
            u = self.upsample(h_g) + u

        x_mu = self.observation_density(u)

        return x_mu
