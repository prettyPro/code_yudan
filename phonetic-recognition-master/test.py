
# -*- coding: utf-8 -*-
# @DATE    : 2017/2/14 17:50
# @Author  :
# @File    : stock_predict.py

import os
import sys
import datetime

import tensorflow as tf
import pandas as pd
import numpy as np
from yahoo_finance import Share
import matplotlib.pyplot as plt

from utils import get_n_day_before, date_2_str


class StockRNN(object):
    def __init__(self, seq_size=12, input_dims=1, hidden_layer_size=12, stock_id="BABA", days=365, log_dir="stock_model/"):
        self.seq_size = seq_size
        self.input_dims = input_dims
        self.hidden_layer_size = hidden_layer_size
        self.stock_id = stock_id
        self.days = days
        self.data = self._read_stock_data()["Adj_Close"].astype(float).values
        self.log_dir = log_dir

    def _read_stock_data(self):
        stock = Share(self.stock_id)
        end_date = date_2_str(datetime.date.today())
        start_date = get_n_day_before(200)
        # print(start_date, end_date)

        his_data = stock.get_historical(start_date=start_date, end_date=end_date)
        stock_pd = pd.DataFrame(his_data)
        stock_pd["Adj_Close"] = stock_pd["Adj_Close"].astype(float)
        stock_pd.sort_values(["Date"], inplace=True, ascending=True)
        stock_pd.reset_index(inplace=True)
        return stock_pd[["Date", "Adj_Close"]]

    def _create_placeholders(self):
        with tf.name_scope(name="data"):
            self.X = tf.placeholder(tf.float32, [None, self.seq_size, self.input_dims], name="x_input")
            self.Y = tf.placeholder(tf.float32, [None, self.seq_size], name="y_input")

    def init_network(self, log_dir):
        print("Init RNN network")
        self.log_dir = log_dir
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restore")

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def _create_rnn(self):
        W = tf.Variable(tf.random_normal([self.hidden_layer_size, 1], name="W"))
        b = tf.Variable(tf.random_normal([1], name="b"))
        with tf.variable_scope("cell_d"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size)
        with tf.variable_scope("rnn_d"):
            outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(self.X)[0], 1, 1])
        out = tf.matmul(outputs, W_repeated) + b
        out = tf.squeeze(out)
        return out

    def _data_prepare(self):
        self.train_x = []
        self.train_y = []
        # data
        data = np.log1p(self.data)
        for i in xrange(len(data) - self.seq_size - 1):
            self.train_x.append(np.expand_dims(data[i: i + self.seq_size], axis=1).tolist())
            self.train_y.append(data[i + 1: i + self.seq_size + 1].tolist())

    def train_pred_rnn(self):

        self._create_placeholders()

        y_hat = self._create_rnn()
        self._data_prepare()
        loss = tf.reduce_mean(tf.square(y_hat - self.Y))
        train_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        feed_dict = {self.X: self.train_x, self.Y: self.train_y}

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in xrange(1, 20001):
                _, loss_ = sess.run([train_optim, loss], feed_dict=feed_dict)
                if step % 100 == 0:
                    print("{} {}".format(step, loss_))
            saver.save(sess, self.log_dir + "model.ckpt")

            # prediction
            prev_seq = self.train_x[-1]
            predict = []
            for i in range(5):
                next_seq = sess.run(y_hat, feed_dict={self.X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
            predict = np.exp(predict) - 1
            print(predict)
            self.pred = predict

    def visualize(self):
        pred = self.pred
        plt.figure()
        plt.legend(prop={'family': 'SimHei', 'size': 15})
        plt.plot(list(range(len(self.data))), self.data, color='b')
        plt.plot(list(range(len(self.data), len(self.data) + len(pred))), pred, color='r')
        plt.title(u"{}股价预测".format(self.stock_id), fontproperties="SimHei")
        plt.xlabel(u"日期", fontproperties="SimHei")
        plt.ylabel(u"股价", fontproperties="SimHei")
        plt.savefig("stock.png")
        plt.show()


if __name__ == "__main__":
    stock = StockRNN()
    # print(stock.read_stock_data())
    log_dir = "stock_model"
    stock.train_pred_rnn()
    stock.visualize()
