import tensorflow as tf
import tensorflow.contrib.summary as summary

from gqn import GenerativeQueryNetwork
from gqn_datasets.data_reader import DataReader
from utils import *

import os

# Training hyper-parameters

mu_i = 5e-4
mu_f = 5e-5
mu_n = 1.6e6

sigma_i = 2.0
sigma_f = 0.7
sigma_n = 4e4  # 2e5

batch_size = 36
test_batch_size = 10
context_size = 5

S_max = int(2e6)

# Model hyper-parameters

im_size = 256

x_dim = 3
r_dim = 64  # 256
h_dim = 64  # 256
z_dim = 32  # 256

L = 12

# Overhead

root = "/localdata/auguste/gqn-dataset"
logs_path = "/localdata/auguste/logs_test"


if __name__ == '__main__':
    session_name = get_session_name()
    session_logs_path = os.path.join(logs_path, session_name)

    global_step = tf.train.get_or_create_global_step()

    train_data_reader = DataReader(
        'shepard_metzler_5_parts', batch_size, context_size, root)
    test_data_reader = DataReader(
        'shepard_metzler_5_parts', test_batch_size,
        context_size, root, mode='test'
    )

    model = GenerativeQueryNetwork(x_dim, r_dim, h_dim, z_dim)

    lr = tf.train.polynomial_decay(mu_i, global_step, mu_n, mu_f)
    sigma_s = tf.train.polynomial_decay(sigma_i, global_step, sigma_n, sigma_f)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    writer = summary.create_file_writer(session_logs_path, max_queue=1)
    writer.set_as_default()

    with summary.record_summaries_every_n_global_steps(500):

        batch = train_data_reader.read()
        x_mu, x_q, r, kl = model(batch)

        output_dist = tf.distributions.Normal(loc=x_mu, scale=sigma_s)
        log_likelihood = tf.reduce_logsumexp(output_dist.log_prob(x_q))

        loss = kl - log_likelihood
        optimize = optimizer.minimize(loss, global_step=global_step)

        output_dist_const_var = tf.distributions.Normal(
            loc=x_mu, scale=sigma_f)
        log_likelihood_const_var = tf.reduce_logsumexp(
            output_dist_const_var.log_prob(x_q))

        summary.scalar(
            "log-likelihood", log_likelihood, family="train")
        summary.scalar(
            "log-likelihood constant variance",
            log_likelihood_const_var, family="train"
        )
        summary.image("inference output", cast_im(x_mu[0:3]), max_images=3)
        summary.image("inference target", cast_im(x_q[0:3]), max_images=3)

        summary.scalar("learning_rate", lr, family="hyper-parameters")
        summary.scalar("sigma_s", sigma_s, family="hyper-parameters")

        # Test set

        batch = test_data_reader.read()
        x_mu, x_q, r = model.sample(batch)

        output_dist = tf.distributions.Normal(loc=x_mu, scale=sigma_s)
        log_likelihood = tf.reduce_logsumexp(output_dist.log_prob(x_q))

        output_dist_const_var = tf.distributions.Normal(
            loc=x_mu, scale=sigma_f)
        log_likelihood_const_var = tf.reduce_logsumexp(
            output_dist_const_var.log_prob(x_q))

        summary.scalar(
            "log-likelihood", log_likelihood, family="test")
        summary.scalar(
            "log-likelihood constant variance",
            log_likelihood_const_var, family="test"
        )

        summary.image("generation output", cast_im(x_mu[0:3]), max_images=3)
        summary.image("generation target", cast_im(x_q[0:3]), max_images=3)

        context = batch.query.context.frames[0:3]
        batch_size, context_size, *x_shape = context.shape
        context = tf.reshape(context, [-1, *x_shape])
        summary.image("generation context", cast_im(context), max_images=5)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        summary.initialize(graph=tf.get_default_graph())

        for s in range(S_max):
            l, *_ = sess.run([loss, optimize, summary.all_summary_ops()])

            if s % 100 == 0:
                print("Iteration: {}  Loss: {}".format(s, l))
