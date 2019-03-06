import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.nn import relu


class Encoder(tf.keras.Model):
    def __init__(self, r_dim):
        super().__init__()
        k = r_dim

        self.c1 = Conv2D(k, 2, strides=2, activation=relu)
        self.c2 = Conv2D(k, 2, strides=2, activation=relu)
        self.c3 = Conv2D(k//2, 3, padding='SAME', activation=relu)
        self.c4 = Conv2D(k, 2, strides=2, activation=relu)

        self.c5 = Conv2D(k, 3, padding='SAME', activation=relu)
        self.c6 = Conv2D(k//2, 3, padding='SAME', activation=relu)
        self.c7 = Conv2D(k, 3, padding='SAME', activation=relu)
        self.c8 = Conv2D(k, 1, padding='SAME', activation=relu)

    def call(self, x, v):
        batch_size, v_dim = v.shape
        batch_size, im_size, im_size, x_dim = x.shape
        broadcast_size = 16

        v = tf.reshape(v, [batch_size, 1, 1, v_dim])
        v = tf.broadcast_to(
            v, [batch_size, broadcast_size, broadcast_size, v_dim])

        skip_in = self.c1(x)
        skip_out = self.c2(skip_in)

        x = self.c3(skip_in)
        x = self.c4(x) + skip_out

        skip_in = tf.concat([x, v], 3)
        skip_out = self.c5(skip_in)

        x = self.c6(skip_in)
        x = self.c7(x) + skip_out

        x = self.c8(x)
        return x
