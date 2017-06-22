import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, args, infer=False):
        self.args = args ###
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        try:
            model_selection = {
                'rnn': tf.contrib.rnn.BasicRNNCell,
                'gru': tf.contrib.rnn.GRUCell,
                'lstm': tf.contrib.rnn.core_rnn_cell.BasicLSTMCell
            }
            cell_fn = model_selection[args.model]
        except KeyError:
            raise Exception(
                "model type not supported: {}".format(args.model))

        cell = cell_fn(args.num_units, state_is_tuple=True)

        stacked_cell = tf.contrib.rnn.MultiRNNCell(
            [cell] * args.num_layers,
            state_is_tuple=True
        )

        if infer is False and args.keep_prob < 1:
            stacked_cell = tf.contrib.rnn.DropoutWrapper(
                stacked_cell, output_keep_prob=args.keep_prob
            )

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
        x_list = tf.split(
            value=input_data,
            num_or_size_splits=args.seq_length,
            axis=1
        )
        # list of [batch_size, 3]
        x_list = [tf.squeeze(x, axis=[1]) for x in x_list]

        # [batch_size x seq_length, 3]
        output_data_flat = tf.reshape(output_data, [-1, 3])
        x1_data, x2_data, eos_data = tf.split(
            value=output_data_flat,
            num_or_size_splits=3,
            axis=1
        )

        # h_list : [batch_size, num_units] x seq_length
        # state_out : [batch_size, cell.state_size]
        h_list, state_out = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=x_list,  # list of [batch_size, 3]
            initial_state=state_in, # [batch_size, cell.state_size]
            cell=stacked_cell,      # [num_units]
            loop_function=None
        )
        # [batch_size x seq_length, num_units]
        h = tf.concat(h_list, 0)
        state_out = tf.identity(state_out, name='state_out')

        # e       # EOS
        # pi      # mixture weights
        # mu      # mean (x1, x2)
        # sigma   # standard deviation (x1, x2)
        # rho     # correlations
        num_param = 1 + args.num_mixture * 6

        with tf.variable_scope('rnn_linear'):
            w = tf.get_variable('w', [args.num_units, dim_y])
            b = tf.get_variable('b', [dim_y])
            y_hat = tf.nn.xw_plus_b(h, w, b, 'y_hat')
            # [batch_size x seq_length, num_param]

        z_eos = y_hat[:, 0]
        y_eos = tf.sigmoid(z_eos)

        z_pi, z_mu1, z_mu2, z_sig1, z_sig2, z_cor = tf.split(
            value=z[:, 1:],
            num_or_size_splits=6,
            axis=1
        ) # [batch_size x seq_length, num_mixture]

