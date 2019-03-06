import tensorflow as tf

from encoder import Encoder
from generator import Generator


class GenerativeQueryNetwork(tf.keras.Model):
    def __init__(self, x_dim, r_dim, h_dim, z_dim, L=12):
        super().__init__()

        self.r_dim = r_dim
        self.encode = Encoder(r_dim)
        self.generate = Generator(x_dim, z_dim, h_dim, L)

    def call(self, batch):
        context_frames = batch.query.context.frames
        context_cameras = batch.query.context.cameras
        batch_size, context_size, *x_shape = context_frames.shape
        batch_size, context_size, *v_shape = context_cameras.shape

        x = tf.reshape(context_frames, [-1, *x_shape])
        v = tf.reshape(context_cameras, [-1, *v_shape])

        phi = self.encode(x, v)

        _, *phi_shape = phi.shape
        phi = tf.reshape(phi, [-1, context_size, *phi_shape])

        r = tf.reduce_sum(phi, axis=1)

        x_q, v_q = batch.target, batch.query.query_camera
        x_mu, kl = self.generate(x_q, v_q, r)

        return x_mu, x_q, r, kl

    def sample(self, batch):
        context_frames = batch.query.context.frames
        context_cameras = batch.query.context.cameras
        batch_size, context_size, *x_shape = context_frames.shape
        batch_size, context_size, *v_shape = context_cameras.shape

        x = tf.reshape(context_frames, [-1, *x_shape])
        v = tf.reshape(context_cameras, [-1, *v_shape])

        phi = self.encode(x, v)

        _, *phi_shape = phi.shape
        phi = tf.reshape(phi, [-1, context_size, *phi_shape])

        r = tf.reduce_sum(phi, axis=1)

        x_q, v_q = batch.target, batch.query.query_camera
        x_mu = self.generate.sample(v_q, r, x_shape[0])

        return x_mu, x_q, r
