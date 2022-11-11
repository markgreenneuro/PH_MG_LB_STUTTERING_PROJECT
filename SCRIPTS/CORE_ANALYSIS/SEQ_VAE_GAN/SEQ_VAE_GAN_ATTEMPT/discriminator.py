import tensorflow as tf

print(tf.__version__)
import numpy as np


class HighWay(tf.keras.layers.Layer):
    def __init__(self, bias_highway=0.2, num_layers=1, *args, **kwargs):
        super(HighWay, self).__init__(*args, **kwargs)

        self.f = tf.nn.relu
        self.bias_highway = bias_highway
        self.num_layers = num_layers

    def build(self, input_shape):
        self.matrix = tf.Variable(tf.zeros([input_shape[0], input_shape[1]]), dtype=tf.float32)
        self.bias_term = tf.Variable(tf.zeros([input_shape[1], input_shape[1]]), dtype=tf.float32)
        return self

    def call(self, input, training=True, *args, **kwarg):
        input = tf.cast(input, tf.float32)
        self.matrix_input = tf.matmul(tf.transpose(input), self.matrix,
                                      transpose_a=False, transpose_b=False) + self.bias_term
        self.g = self.f(self.matrix_input)
        self.t = tf.sigmoid(self.matrix_input) + self.bias_highway

        output = (self.t * (self.g + (float(1) - self.t)) @ tf.transpose(input))

        return output


class DiscriminationLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedded_size, *args, **kwargs):
        super(DiscriminationLayer, self).__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embedded_size = embedded_size

    def call(self, input, training=True, *args, **kwargs):
        self.W = tf.compat.v1.random_uniform([self.vocab_size, self.embedded_size], -1.0, 1.0)
        embedded_chars = tf.compat.v1.nn.embedding_lookup(self.W, input)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        return embedded_chars_expanded


class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, filter_size, embedded_size, embedded_chars_expanded, num_filter, num_classes,
                 sequence_length, *args, **kwargs):
        super(ConvolutionLayer, self).__init__(*args, **kwargs)
        self.filter_size = filter_size
        self.embedded_size = embedded_size
        self.embedded_chars_expanded = embedded_chars_expanded
        self.num_filter = num_filter
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.filter_shape = [self.filter_size, self.embedded_size, 1, self.num_filter]


    def call(self, input, training=True, *args, **kwargs):
        self.W = tf.compat.v1.truncated_normal(self.filter_shape, stddev=0.1)
        self.b = tf.constant(0.1, shape=[self.num_filter])

        self.conv = tf.compat.v1.nn.conv2d(self.embedded_chars_expanded, self.W,
                                           strides=[1, 1, 1, 1],
                                           padding="VALID",
                                           name="conv")
        # Apply nonlinearity
        self.h = tf.nn.relu(tf.nn.bias_add(self.conv, self.b), name="relu")
        # Maxpooling over the outputs

        output = tf.compat.v1.nn.max_pool(self.h, ksize=[1, self.sequence_length - self.filter_size + 1, 1, 1],
                                          strides=[1, 1, 1, 1],
                                          padding='VALID',
                                          name="pool")
        return output


class GetScores(tf.keras.layers.Layer):
    def __init__(self, num_filters, num_classes, *args, **kwargs):
        super(GetScores, self).__init__(*args, **kwargs)
        self.num_filters = num_filters
        self.num_classes = num_classes

    def call(self, input, training=True, *args, **kwargs):
        self.W = tf.compat.v1.truncated_normal([self.num_filters, self.num_classes], stddev=0.1)
        self.b = tf.constant(0.1, shape=[self.num_classes])
        scores = tf.transpose(
            tf.matmul(tf.transpose(self.W), tf.squeeze(input), transpose_a=False, transpose_b=False)) + self.b
        return scores


class Discriminator(tf.keras.Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedded_size, filter_sizes, num_filters,
                 *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedded_size = embedded_size
        self.filter_sizes = filter_sizes
        self.dropout_keep_prob = tf.Variable(0, name="dropout_keep_prob")

    def build(self, input_shape):
        self.output_size = input_shape[0]
        self.input_size = input_shape[1]

        self.discrimination_layer = DiscriminationLayer(vocab_size=self.vocab_size,
                                                        embedded_size=self.embedded_size)

        self.h_highway = HighWay()

        self.get_scores = GetScores(num_filters=self.num_filters, num_classes=self.num_classes)

        self.dense = tf.keras.layers.Dense(2, kernel_regularizer='l2')
        return self

    def call(self, input, training=True, *args, **kwarg):
        embedded_chars_expanded = self.discrimination_layer(input)
        self.convol_layer = ConvolutionLayer(filter_size=self.filter_sizes,
                                             embedded_size=self.embedded_size,
                                             embedded_chars_expanded=embedded_chars_expanded,
                                             num_classes=self.num_classes,
                                             num_filter=self.num_filters,
                                             sequence_length=self.sequence_length)
        conv = self.convol_layer(input)
        print('conv')
        h_drop = self.h_highway(conv)
        print('h_drop')
        scores = self.get_scores(h_drop)
        print('scores')
        regularized_scores = self.dense(scores)
        regularized_scores = tf.squeeze(tf.argmax(regularized_scores, 1))
        regularized_scores = tf.one_hot(regularized_scores, 2)
        print('regularized_scores')
        return regularized_scores


discriminator = Discriminator(
    sequence_length=8,
    num_classes=2, vocab_size=5000,
    embedded_size=64,
    filter_sizes=2,
    num_filters=200)
from dataloader import Dis_dataloader

dis_data_loader = Dis_dataloader(seq_len=8, batch_size=100)
data = dis_data_loader.load_train_data('real_data_8_numpy', 'generator_sample')



# /home/mda/anaconda3/envs/SeqVAEGAN/lib/python3.7/site-packages
d_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
losses = tf.nn.softmax_cross_entropy_with_logits

discriminator.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=d_optimizer,
                      run_eagerly=False)
discriminator.fit(data.sentences, data.labels, batch_size=100)
