import argparse
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf

from utils import DataLoader
from model import Model

FLAGS = None

# do not print warning logs
# https://stackoverflow.com/questions/35911252
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(_):

    if os.path.exists(FLAGS.model_dir) is False:
        os.makedirs(FLAGS.model_dir)

    config_fname = os.path.join(FLAGS.model_dir, 'config.pkl')
    with open(config_fname, 'wb') as f:
        pickle.dump(FLAGS, f)

    data_loader = DataLoader()
    model = Model(FLAGS)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # saver = tf.train.Saver(tf.global_variables())

        for e in range(FLAGS.num_epoch):
            data_loader.reset_batch_pointer()
            vx, vy = data_loader.validation_data()
            feed_dict = {
                model.input_data: v_x,
                model.target_data: v_y,
                model.state_in: model.state_in.eval()
            }
            state = model.state_in.eval()

            # for b in range(data_loader.num_batches):

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', type=str, default='./model',
        help='directory to save model to')

    parser.add_argument(
        '--batch_size', type=int, default=50,
        help='minibatch size')
    parser.add_argument(
        '--num_epoch', type=int, default=30,
        help='number of epochs')
    parser.add_argument(
        '--learning_rate', type=float, default=0.005,
        help='learning rate')
    parser.add_argument(
        '--keep_prob', type=float, default=0.8,
        help='dropout keep probability')

     parser.add_argument(
        '--model', type=str, default='lstm',
        help='rnn, gru, or lstm')
     parser.add_argument(
        '--num_units', type=int, default=256,
        help='size of RNN hidden state')
     parser.add_argument(
        '--num_layers', type=int, default=2,
        help='number of layers in the RNN')
    parser.add_argument(
        '--seq_length', type=int, default=300,
        help='RNN sequence length')

    parser.add_argument(
        '--save_every', type=int, default=500,
        help='save frequency')
    parser.add_argument(
        '--grad_clip', type=float, default=10.,
        help='clip gradients at this value')

    parser.add_argument(
        '--decay_rate', type=float, default=0.95,
        help='decay rate for rmsprop')
    parser.add_argument(
        '--num_mixture', type=int, default=20,
        help='number of gaussian mixtures')
    parser.add_argument(
        '--data_scale', type=float, default=20,
        help='factor to scale raw data down by')
    FLAGS, unparsed = parser.parse_known_args()
    print([sys.argv[0]], FLAGS, unparsed)
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
