import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, args, infer=False):
        if infer is True:
            args.batch_size = 1
            args.seq_length = 1

        # input
        with tf.name_scope('input'):
            input_data = tf.placeholder(
                shape=[None, args.seq_length, 3],
                dtype=tf.float32,
                name='data_in'
            )
            output_data = tf.placeholder(
                shape=[None, args.seq_length, 3],
                dtype=tf.float32,
                name='data_out'
            )
            zero_state = stacked_cell.zero_state(
                batch_size=args.batch_size,
                dtype=tf.float32
            )
            state_in = tf.identity(zero_state, name='state_in')

        # list of [batch_size, 1, 3]
        input_list = tf.split(
            value=input_data,
            num_or_size_splits=args.seq_length,
            axis=1
        )
        # list of [batch_size, 3]
        input_list = [tf.squeeze(x, axis=[1]) for x in x_list]

        # model selection
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicRNNCell
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/GRUCell
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell
        try:
            model_selection = {
                'rnn': tf.contrib.rnn.BasicRNNCell,
                'gru': tf.contrib.rnn.GRUCell,
                'lstm': tf.contrib.rnn.BasicLSTMCell
            }
            cell_fn = model_selection[args.model]
            cell = cell_fn(args.num_units)
        except KeyError:
            raise Exception(
                "model type not supported: {}".format(args.model)
            )

        stacked_cell = tf.contrib.rnn.MultiRNNCell(
            [cell] * args.num_layers,
            state_is_tuple=True
        )
        if infer is False and args.keep_prob < 1:
            stacked_cell = tf.contrib.rnn.DropoutWrapper(
                stacked_cell,
                output_keep_prob=args.keep_prob
            )
        # h_list : [batch_size, num_units] x seq_length
        # state_out : [batch_size, cell.state_size]
        h_list, state_out = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=input_list,  # list of [batch_size, 3]
            initial_state=state_in, # [batch_size, cell.state_size]
            cell=stacked_cell,      # [num_units x num_layers]
            loop_function=None
        )
        # [batch_size x seq_length, num_units]
        h = tf.concat(h_list, 0)
        # h = tf.reshape(output, [-1, args.num_units])
        state_out = tf.identity(state_out, name='state_out')

        # e       # EOS
        ############################
        # pi      # mixture weights
        # mu      # mean (x1, x2)
        # sigma   # standard deviation (x1, x2)
        # rho     # correlations
        num_param = 1 + args.num_mixture * 6

        with tf.variable_scope('rnn_linear'):
            w = tf.get_variable('w', [args.num_units, num_param])
            b = tf.get_variable('b', [num_param])
            param_hat = tf.nn.xw_plus_b(h, w, b, 'param_hat')
            # [batch_size x seq_length, num_param]

        x1_data, x2_data, eos_data = tf.split(
            value=tf.reshape(output_data),
            num_or_size_splits=3,
            axis=1
        )

        def get_loss(target_x1, target_x2, target_eos, param_hat):
            """
            input
                - x1, x2, eos: [batch_size x seq_length, 1]
                - param_hat: [batch_size x seq_length, num_param]
            output
                - Sum of batch_size x seq_length NLLs
            """

            # split param_hat -> [batch_size x seq_length, num_mixture]
            z_eos = param_hat[:, 0]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_rho = tf.split(
                value=z[:, 1:],
                num_or_size_splits=6,
                axis=1
            )

            # get params
            eos = tf.sigmoid(z_eos)
            pi = tf.nn.softmax(z_pi)
            sigma1, sigma2 = tf.exp(z_sigma1), tf.exp(z_sigma2)
            rho = tf.tanh(z_rho)

            # get frequently used terms
            n1 = tf.subtract(target_x1, mu1)
            n2 = tf.subtract(target_x2, mu2)
            s1s2 = tf.multiply(sigma1, sigma2)
            drho = 1 - tf.square(rho)

            # get likelihood
            z = tf.square(tf.div(n1, sigma1)) + tf.square(tf.div(n2, sigma2))
            z -= 2 * tf.div(
                tf.multiply(rho, tf.multiply(n1, n2)), s1s2
            )
            denom = 2 * tf.pi * tf.multiply(s1s2, tf.sqrt(drho))
            normal_prob = tf.div(tf.exp(tf.div(-z, 2 * drho)), denom)
            mixture_normal_prob = tf.reduce_sum(
                tf.multiply(pi, normal_prob),
                axis=1,
                keep_dims=True
            )

            # get negative log likelihood
            epsilon = 1e-20
            nll_x1x2 = -tf.log(tf.maximum(mixture_normal_prob, epsilon))
            nll_eos = -tf.log(
                tf.multiply(z_eos, target_eos) + tf.multiply(1-z_eos, 1-target_eos)
            )
            return tf.reduce_sum(nll_x1x2 + nll_eos)

        # [batch_size, seq_length, num_mixture]


