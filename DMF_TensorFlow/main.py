import os
import json
import math
from progressbar import ProgressBar
import configparser
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import data
import evaluation
from DMF import DMF


def train(result_dir, model, data_splitter, train_data, validation_data, batch_size, config):
    epoch_data = []
    best_ndcg = 0
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
            hit_ratio, ndcg = evaluation.evaluate(model, sess, validation_data, config.getint('EVALUATION', 'top_k'))
            epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
            if ndcg > best_ndcg:
                tf.train.Saver().save(sess, os.path.join(result_dir, 'model'))
            print('\n[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    return epoch_data


def get_feed_dict(model, train_data, start, end):
    feed_dict = {}
    feed_dict[model.user] = train_data[start:end, 0]
    feed_dict[model.item] = train_data[start:end, 1]
    feed_dict[model.rating] = train_data[start:end, 2]
    return feed_dict


def save_train_result(result_dir, epoch_data):
    with open(os.path.join(result_dir, 'epoch_data.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)


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
            epoch_data = train(result_dir, model, data_splitter, train_data, validation_data, batch_size, config)
            save_train_result(result_dir, epoch_data)


if __name__ == "__main__":
    main()