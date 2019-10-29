import tensorflow as tf
import tflib.ops.linear
import tflib as lib
import tensorflow.contrib as tc

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Generator(object):
    def __init__(self, z_dim=15, x_dim=512):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'ami/clus_wgan/g_net'

    def __call__(self, z, keep=1.0):

        with tf.variable_scope(self.name) as vs:
            output = lib.ops.linear.Linear('Generator.Input', self.z_dim, 512, z)
            output = tf.nn.relu(output)
            #output = leaky_relu(output)

            output = lib.ops.linear.Linear('Generator.L1', 512, 512, output)
            output = tf.nn.relu(output)
            #output = leaky_relu(output)

            output = lib.ops.linear.Linear('Generator.L2', 512, self.x_dim, output)
            return tf.reshape(output, [-1, self.x_dim])

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator(object):
    def __init__(self, x_dim=512):
        self.x_dim = x_dim
        self.name = 'ami/clus_wgan/d_net'

    def __call__(self, x, keep=1.0):
        with tf.variable_scope(self.name) as vs:
            #if reuse:
                #vs.reuse_variables()

            output = lib.ops.linear.Linear('Discriminator.Input', self.x_dim, 512, x)
            output = tf.nn.relu(output)
            #output = leaky_relu(output)

            output = lib.ops.linear.Linear('Discriminator.L1', 512, 512, output)
            output = tf.nn.relu(output)
            #output = leaky_relu(output)

            output = lib.ops.linear.Linear('Discriminator.L2', 512, 512, output)
            output = tf.nn.relu(output)
            #output = leaky_relu(output)

            output = lib.ops.linear.Linear('Discriminator.L3', 512, 1, output)
            return tf.reshape(output, [-1])

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):

    def __init__(self, z_dim=15, dim_gen=5, x_dim=512):
        self.z_dim = z_dim
        self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = 'ami/clus_wgan/enc_net'

    def __call__(self, x, keep=1.0):
        with tf.variable_scope(self.name) as vs:
            # if reuse:
                #vs.reuse_variables()
            output = lib.ops.linear.Linear('Encoder.Input', self.x_dim, 512, x)
            output = tf.nn.relu(output)
            #output = leaky_relu(output)

            output = lib.ops.linear.Linear('Encoder.L1', 512, 512, output)
            output1 = tf.nn.relu(output)
            #output1 = leaky_relu(output)
            tf.add_to_collection("output1", output1)

            output = lib.ops.linear.Linear('Encoder.L2', 512, self.z_dim, output1)
            tf.add_to_collection("output", output)

            logits = output[:, self.dim_gen:]
            tf.add_to_collection("logits", logits)
            y = tf.nn.softmax(logits)
            tf.add_to_collection("y", y)

            return output[:, 0:self.dim_gen], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


