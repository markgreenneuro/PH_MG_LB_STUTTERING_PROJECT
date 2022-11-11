from pathlib import Path

import tensorflow as tf


class HighWay(tf.keras.layers.Layer):
    def __init__(self, bias_highway=0.2, num_layers=1, *args, **kwargs):
        super(HighWay, self).__init__(*args, **kwargs)

        self.f = tf.nn.relu
        self.bias_highwy = bias_highway
        self.num_layers = num_layers

    def build(self, input_shape):
        self.matrix = tf.Variable(tf.zeros([input_shape[0], input_shape[1]]), dtype=tf.float32)
        self.bias_term = tf.Variable(tf.zeros([input_shape[1], input_shape[1]]), dtype=tf.float32)
        return self

    def call(self, input, *args, **kwargs):
        input = tf.cast(input, tf.float32)
        self.matrix_input = tf.matmul(tf.transpose(input), self.matrix,
                                      transpose_a=False, transpose_b=False) + self.bias_term

        for idx in range(self.num_layers):
            self.g = self.f(self.matrix_input)
            self.t = tf.sigmoid(self.matrix_input) + self.bias_highway
            output = (self.t * (self.g + (float(1) - self.t)) @ tf.transpose(input))
            input = output
            return output


class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, sequence_length, filter_size, num_filter, num_layer, output_size, input_size,
                 vocab_size, embedding_size,
                 *args, **kwargs):
        super(ConvolutionLayer, self).__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.num_layer = num_layer
        self.output_size = output_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.W = tf.Variable(tf.compat.v1.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
        self.pooled_outputs = []
        self.intializers = tf.initializers.truncated_normal(stddev=0.1)

    def call(self, input, *args, **kwargs):
        self.embedded_chars = tf.compat.v1.nn.embedding_lookup(self.W, input)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        filter_shape = [self.filter_size, self.embedding_size, 1, self.num_filter]
        self.W = tf.Variable(self.intializers(filter_shape), name="W")
        self.b = tf.Variable(tf.keras.backend.repeat_elements(tf.constant([0.1]), self.num_filter, axis=0),
                             shape=self.num_filter,
                             name="b")

        self.conv = tf.compat.v1.nn.conv2d(self.embedded_chars_expanded, self.W,
                                           strides=[1, 1, 1, 1],
                                           padding="VALID",
                                           name="conv")

        # Apply nonlinearity
        self.h = tf.nn.relu(tf.nn.bias_add(self.conv, self.b), name="relu")

        # Maxpooling over the outputs
        output = tf.compat.v1.nn.max_pool(self.h,
                                          ksize=[1, self.sequence_length - self.filter_size + 1, 1, 1],
                                          strides=[1, 1, 1, 1],
                                          padding='VALID',
                                          name="pool")
        return output



class GetScores(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(GetScores, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.W = tf.Variable(
            tf.compat.v1.truncated_normal([int(input_shape[0]), int(input_shape[1])], stddev=0.1), dtype=tf.float32,
            name="W")
        self.b = tf.Variable([int(input_shape[1])], dtype=tf.float32, name="b")
        return self

    def call(self, input, training=False, *args, **kwargs):
        scores = tf.matmul(tf.transpose(self.W), input, transpose_a=False, transpose_b=False) + self.b
        output = tf.cast(tf.argmax(scores, 1, name="predictions"), dtype=tf.float32)
        return output


class CalculateLoss(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CalculateLoss, self).__init__(*args, **kwargs)

    def call(self, input, *args, **kwargs):
        y_logits = input[0]
        y_labels = input[1]
        y_labels = tf.cast(tf.argmax(y_labels, 1), dtype=tf.float32)
        y_logits = tf.expand_dims(y_logits, -1)
        y_labels = tf.expand_dims(y_labels, -1)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_labels)
        losses = tf.expand_dims(losses, -1)

        output = tf.reduce_mean(losses, axis=0)

        return output


class Discriminator(tf.keras.Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 learning_rate, *args, **kwargs):
        super(Discriminator, self).__init__(dynamic=True, *args, **kwargs)
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.learning_rate = learning_rate
        self.dropout_keep_prob = tf.Variable(0, name="dropout_keep_prob")

    def build(self, *args, **kwargs):
        self.convol_layer = ConvolutionLayer(sequence_length=self.sequence_length, filter_size=self.filter_sizes,
                                             num_filter=self.num_filters,
                                             output_size=self.output_size, input_size=self.input_size,
                                             vocab_size=self.vocab_size, embedding_size=self.embedding_size)
        self.h_highway = HighWay()

        self.get_scores = GetScores()

        self.dense = tf.keras.layers.Dense(1, kernel_regularizer='l2')
        return self

    def call(self, input, *args, **kwargs):
        input_conv_layer = self.convol_layer(input[0])
        h_pool_flat = tf.reshape(input_conv_layer, [-1, 4])
        h_drop = self.highway(h_pool_flat)
        scores = self.get_scores([h_drop, input[1]])
        scores = tf.expand_dims(scores, -1)
        output = self.dense(scores)
        return output


from dataloader import Dis_dataloader

dis_data_loader = Dis_dataloader(seq_len=8, batch_size=100)
data = dis_data_loader.load_train_data('real_data_8', 'generator_sample')
