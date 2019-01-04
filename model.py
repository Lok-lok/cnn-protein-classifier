import tensorflow as tf
import csv
import os
    
def model_fn(features, labels, mode):
    labels = tf.cast(labels, tf.int32)
    input_layer = tf.reshape(features, [-1, 512, 512, 4])

    conv1 = tf.layers.conv2d(inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)
        
    pool1 = tf.layers.max_pooling2d(inputs = conv1, 
        pool_size = [4, 4], 
        strides = 2)
        
    conv2 = tf.layers.conv2d(inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2, 
        pool_size = [4, 4], 
        strides = 2)
        
    after_conv_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

    dense1 = tf.layers.dense(inputs = after_conv_flat, 
        units = 2048, 
        activation = tf.nn.relu)
        
    dense2 = tf.layers.dense(inputs = dense1, 
        units = 128, 
        activation = tf.nn.relu)
        
    # dropout
    
    logits = tf.layers.dense(inputs = dense2, units = 2)
    
    argmax = tf.argmax(input = logits, axis = 1)
    
    softmax = tf.nn.softmax(logits, name = "softmax_tensor")
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
    accuracy = tf.metrics.accuracy(labels = labels, predictions = argmax)
    
    log = {"accuracy" : accuracy[1], "loss" : loss}
    logging_hook = tf.train.LoggingTensorHook(tensors = log, every_n_iter = 50)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = softmax)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, training_hooks = [logging_hook])
        
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy" : accuracy}
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
