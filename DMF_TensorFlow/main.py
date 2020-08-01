import os
import json
import math
from progressbar import ProgressBar
import configparser
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import data
from DMF import DMF


def train(result_dir, model, data_splitter, train_data, validation_data, batch_size, config):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.getint('MODEL', 'epoch')):
            start = 0
            total_loss = 0
            np.random.shuffle(train_data)
            pb = ProgressBar(1, math.ceil(len(train_data)/batch_size))
            while start < len(train_data):
                _, loss = model.train(
                    sess, get_feed_dict(model, train_data, start, start + batch_size))
                start += batch_size
                total_loss += loss
                pb.update(start // batch_size)
            print('\n[Epoch {}] Loss = {:.2f}'.format(epoch, total_loss))


def get_feed_dict(model, train_data, start, end):
    feed_dict = {}
    feed_dict[model.user] = train_data[start:end, 0]
    feed_dict[model.item] = train_data[start:end, 1]
    feed_dict[model.rating] = train_data[start:end, 2]
    return feed_dict


def main():
    config = configparser.ConfigParser()
    config.read('DMF_TensorFlow/config.ini')

    data_splitter = data.DataSplitter()
    train_data = data_splitter.make_train_data(config.getint('MODEL', 'n_negative'))
    validation_data = data_splitter.make_evaluation_data('validation')
    test_data = data_splitter.make_evaluation_data('test')
    rating_matrix = data_splitter.rating_matrix

    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            print('batch_size = {}, lr = {}'.format(batch_size, lr))
            result_dir = "data/train_result/batch_size_{}-lr_{}-epoch_{}-n_negative_{}-top_k_{}".format(
                batch_size, lr, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
            os.makedirs(result_dir, exist_ok=True)
            tf.reset_default_graph()
            model = DMF(data_splitter.n_user, data_splitter.n_item, rating_matrix, lr, config)
            train(result_dir, model, data_splitter, train_data, validation_data, batch_size, config)


if __name__ == "__main__":
    main()