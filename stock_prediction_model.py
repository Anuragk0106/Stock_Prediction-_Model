import pandas as pd
import tensorflow as tf
import numpy as np

class StockPredictionModel:
    def __init__(self):
        self.LEARNING_RATE = 0.1
        self.NUM_EPOCHS = 100
        self.session = tf.compat.v1.Session()
        self.tf_prediction_graph()

    def tf_prediction_graph(self):
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, name='x')
        self.W = tf.Variable([.1], name='W')
        self.b = tf.Variable([.1], name='b')
        self.y = self.W * self.x + self.b
        self.y_predicted = tf.compat.v1.placeholder(tf.float32, name='y_predicted')
        self.loss = tf.reduce_sum(tf.square(self.y - self.y_predicted))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        self.session.run(tf.compat.v1.global_variables_initializer())

    def load_stock_data(self, stock_name, num_data_points):
        data = pd.read_csv(stock_name,
                           skiprows=0,
                           nrows=num_data_points,
                           usecols=['Price', 'Open', 'Vol.'])
        final_prices = data['Price'].astype(str).str.replace(',', '').astype(float)
        opening_prices = data['Open'].astype(str).str.replace(',', '').astype(float)
        volumes = data['Vol.'].str.strip('MK').astype(float)
        return final_prices, opening_prices, volumes

    def calculate_price_differences(self, final_prices, opening_prices):
        price_differences = []
        for d_i in range(len(final_prices) - 1):
            price_difference = opening_prices[d_i + 1] - final_prices[d_i]
            price_differences.append(price_difference)
        return price_differences

    def calculate_accuracy(self, expected_values, actual_values):
        num_correct = 0
        for a_i in range(len(actual_values)):
            if actual_values[a_i] < 0 < expected_values[a_i]:
                num_correct += 1
            elif actual_values[a_i] > 0 > expected_values[a_i]:
                num_correct += 1
        return (num_correct / len(actual_values)) * 100

    def train_and_predict(self, train_data_file, test_data_file):
        train_final_prices, train_opening_prices, train_volumes = self.load_stock_data(train_data_file, 266)
        train_price_differences = self.calculate_price_differences(train_final_prices, train_opening_prices)
        train_volumes = train_volumes[:-1]

        test_final_prices, test_opening_prices, test_volumes = self.load_stock_data(test_data_file, 22)
        test_price_differences = self.calculate_price_differences(test_final_prices, test_opening_prices)
        test_volumes = test_volumes[:-1]

        for _ in range(self.NUM_EPOCHS):
            self.session.run(self.optimizer, feed_dict={self.x: train_volumes, self.y_predicted: train_price_differences})

        results = self.session.run(self.y, feed_dict={self.x: test_volumes})
        accuracy = self.calculate_accuracy(test_price_differences, results)
        return accuracy
