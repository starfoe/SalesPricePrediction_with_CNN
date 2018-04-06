import numpy as np
import scipy as sp
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import cross_validation

from regression_tools.plotting_tools import (
    plot_partial_depenence,
    predicteds_vs_actuals)
from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


def date_process_Y(datestr):
    [month,day,year] = datestr.split(' ')[0].split('/')
    return int(year)

def date_process_M(datestr):
    [month,day,year] = datestr.split(' ')[0].split('/')

    return int(month)

def date_process_D(datestr):
    [month,day,year] = datestr.split(' ')[0].split('/')
    return int(day)


def seasonal_dummie(df):
   
    def date_process_M(datestr):
        [month,day,year] = datestr.split(' ')[0].split('/')
        return int(month)

    def date_season(int_month):
        if int_month >=1 and int_month < 4:
            return 'First'
        elif int_month >=4 and int_month < 7:
            return 'Second'
        elif int_month >=7 and int_month < 10:
            return 'Third'
        elif int_month >=10 and int_month < 12:
            return 'Fourth'

    df_tmp = df.copy()
    if 'month' in df_tmp.columns:
        month_data = df_tmp['month']
    else:
        df_tmp['month'] = df_tmp['saledate'].apply(date_process_M)      
    df_tmp['season'] = df_tmp['month'].apply(date_season)
    dummies = pd.get_dummies(df_tmp['season'])

    return dummies



if __name__=='__main__':
    
    output_file_path = sys.argv[1]
    
    #Loading the training data
    train_data_full = pd.read_csv('data/Train.zip',compression = 'zip')
    train_data = train_data_full[train_data_full.index %8 ==0] #smaller dataset for the sake of speed
    train_y = train_data['SalePrice'].values
    test_data = pd.read_csv('data/Test.zip',compression = 'zip')
    
    train_data['year'] = train_data['saledate'].apply(date_process_Y)
    test_data['year'] = test_data['saledate'].apply(date_process_Y)
#     train_x['month'] = train_x['saledate'].apply(date_process_M)  

    #feature pipelines
    YearMadeD_fit = Pipeline([
        ('YearMade', ColumnSelector(name='YearMade')),
    ])
    ModelID_fit = Pipeline([
        ('ModelID', ColumnSelector(name='ModelID')),
    ])
    Year_fit = Pipeline([
        ('year', ColumnSelector(name='year')),
    ])

    SalesID_fit = Pipeline([
        ('SalesID', ColumnSelector(name='SalesID')),
    ])
    # First_fit = Pipeline([
    #     ('First', ColumnSelector(name='First')),
    # ])
    # Second_fit = Pipeline([
    #     ('Second', ColumnSelector(name='Second')),
    # ])

    # Fourth_fit = Pipeline([
    #     ('Fourth', ColumnSelector(name='Fourth')),
    # ])
    # Year_fit_spline = Pipeline([
    #     ('Year', ColumnSelector(name='year')),
    #     ('finite_add_spline', NaturalCubicSpline(knots=[1991,1992,1994,1995,1996,1998,2003,2006]))
    # ])

    feature_pipeline = FeatureUnion([
        ('YearMadeD_fit', YearMadeD_fit),
        ('ModelID_fit', ModelID_fit),
        ('Year_fit', Year_fit),
    #     ('Fourth_fit', Fourth_fit),
    #     ('Second_fit', Second_fit),
    #     ('First_fit', First_fit),
        ('SalesID_fit',SalesID_fit)
    ])


    #Data preparation
    feature_pipeline.fit(train_data)
    train_features_df = feature_pipeline.transform(train_data)

    feature_pipeline.fit(test_data)
    test_features_df = feature_pipeline.transform(test_data)

    x_x = train_features_df.values
    y_y = np.reshape(train_y, [train_y.shape[0], 1])
    y_y.astype(float)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    x_x, y_y, test_size=0.2, random_state=42)

    total_len = X_train.shape[0]

    # Parameters
    learning_rate = 0.002
    training_epochs = 500
    batch_size = 10
    display_step = 50
    dropout_rate = 0.9
    # Network Parameters
    n_hidden_1 = 32 # 1st layer number of features
    n_hidden_2 = 200 # 2nd layer number of features
    n_hidden_3 = 200
    n_hidden_4 = 256
    n_input = X_train.shape[1]
    n_classes = 1

    # tf Graph input
    x = tf.placeholder("float", [None, X_train.shape[1]])
    y = tf.placeholder("float", [None, 1])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.square(pred-y))
    # cost = tf.sqrt(tf.reduce_mean(tf.square(tf.log(pred+1)-tf.log(y+1))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(total_len/batch_size)
            # Loop over all batches
            for i in range(total_batch-1):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = Y_train[i*batch_size:(i+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

                # sample prediction
                label_value = batch_y
                estimate = p

            # Display logs per epoch step
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
                print ("[*]----------------------------")
                for i in range(10):
                    print ("label value:", label_value[i], \
                        "estimated value:", estimate[i])
                print ("[*]============================")

        print ("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy

        predicted_vals = sess.run(pred, feed_dict={x: X_test})
        test_features = test_features_df.values
        predicted_val_2 = sess.run(pred, feed_dict={x: test_features})
        accuracy = sess.run(cost, feed_dict={x:X_test, y: Y_test})
    #     print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
        print ("Accuracy:", accuracy)


    df_result_output = pd.DataFrame(test_data['SalesID'],columns = ['SalesID'])
    df_result_output['SalePrice'] = predicted_val_2.reshape(-1,1)
    df_result_output.to_csv(output_file_path)
