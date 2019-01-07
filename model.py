import tensorflow as tf

def bottleneck_block(inputs, filters1, filters2, filters3, strides = (1, 1), kernel_size = (3, 3)):
    conv1 = tf.layers.conv2d(inputs = inputs,
        filters = filters1,
        kernel_size = (1, 1),
        strides = strides)
    conv1_bn = tf.layers.batch_normalization(conv1)
    conv1_fin = tf.nn.relu(conv1_bn)
    
    conv2 = tf.layers.conv2d(inputs = conv1_fin,
        filters = filters2,
        kernel_size = kernel_size,
        padding = "same")
    conv2_bn = tf.layers.batch_normalization(conv2)
    conv2_fin = tf.nn.relu(conv2_bn)
    
    conv3 = tf.layers.conv2d(inputs = conv2_fin,
        filters = filters3,
        kernel_size = (1, 1))
    conv3_bn = tf.layers.batch_normalization(conv3)
    
    shortcut = inputs
    shortcut_bn = tf.layers.batch_normalization(shortcut)
    if tf.shape(inputs) != tf.shape(conv3_bn):
        shortcut = tf.layers.conv2d(inputs = inputs,
            filters = filters3,
            kernel_size = (1, 1),
            strides = strides)
        shortcut_bn = tf.layers.batch_normalization(shortcut)
        
    return tf.nn.relu(conv3_bn + shortcut_bn)
    
def resnet_50_model_fn(features, labels, mode):
    if type(features) is dict:
        features = features['features']
    inputs = tf.reshape(features, [-1, 224, 224, 4])

    conv1 = tf.layers.conv2d(inputs = inputs,
        filters = 64,
        kernel_size = (7, 7),
        strides = (2, 2),
        padding = "same")
    conv1_bn = tf.layers.batch_normalization(conv1)
    conv1_fin = tf.nn.relu(conv1_bn)
        
    max_pool = tf.layers.max_pooling2d(inputs = conv1_fin,
        pool_size = (3, 3),
        strides = (2, 2),
        padding = "same")
        
    conv2 = bottleneck_block(max_pool, 64, 64, 256, strides = (1, 1), kernel_size = (3, 3))
    for i in range(2):
        conv2 = bottleneck_block(conv2, 64, 64, 256, strides = (1, 1), kernel_size = (3, 3))
        
    conv3 = bottleneck_block(conv2, 128, 128, 512, strides = (2, 2), kernel_size = (3, 3))
    for i in range(3):
        conv3 = bottleneck_block(conv3, 128, 128, 512, strides = (1, 1), kernel_size = (3, 3))
        
    conv4 = bottleneck_block(conv3, 256, 256, 1024, strides = (2, 2), kernel_size = (3, 3))
    for i in range(5):
        conv4 = bottleneck_block(conv4, 256, 256, 1024, strides = (1, 1), kernel_size = (3, 3))
        
    conv5 = bottleneck_block(conv4, 512, 512, 2048, strides = (2, 2), kernel_size = (3, 3))
    for i in range(2):
        conv5 = bottleneck_block(conv5, 512, 512, 2048, strides = (1, 1), kernel_size = (3, 3))
        
    avg_pool = tf.layers.average_pooling2d(inputs = conv5,
        pool_size = (7, 7),
        strides = (7, 7))
        
    flat = tf.layers.flatten(avg_pool)
    
    fc = tf.layers.dense(inputs = flat,
        units = 1000,
        activation = tf.nn.relu)
        
    logits = tf.layers.dense(inputs = fc, units = 28)
    
    sigmoid = tf.nn.sigmoid(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = sigmoid)
        
    labels = tf.cast(labels, tf.int32)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
        
    # if mode == tf.estimator.ModeKeys.EVAL:
        # accuracy = tf.metrics.accuracy(labels = labels, predictions = argmax)
        # eval_metric_ops = {"accuracy" : accuracy}
        # tf.summary.scalar('accuracy', accuracy[1])
        # return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def cnn_model_fn(features, labels, mode):
    if type(features) is dict:
        features = features['features']
    input_layer = tf.reshape(features, [-1, 512, 512, 4])

    conv1 = tf.layers.conv2d(inputs = input_layer,
        filters = 32,
        kernel_size = [4, 4],
        padding = "same",
        activation = tf.nn.relu)
        
    pool1 = tf.layers.max_pooling2d(inputs = conv1, 
        pool_size = [4, 4], 
        strides = 4)
        
    conv2 = tf.layers.conv2d(inputs = pool1,
        filters = 64,
        kernel_size = [4, 4],
        padding = "same",
        activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2, 
        pool_size = [4, 4], 
        strides = 4)
        
    after_conv_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

    dense1 = tf.layers.dense(inputs = after_conv_flat, 
        units = 2048, 
        activation = tf.nn.relu)
        
    dense2 = tf.layers.dense(inputs = dense1, 
        units = 128, 
        activation = tf.nn.relu)
        
    # dropout
    
    logits = tf.layers.dense(inputs = dense2, units = 28)
    
    sigmoid = tf.nn.sigmoid(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = sigmoid)
        
    labels = tf.cast(labels, tf.int32)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
        
    # if mode == tf.estimator.ModeKeys.EVAL:
        # accuracy = tf.metrics.accuracy(labels = labels, predictions = argmax)
        # eval_metric_ops = {"accuracy" : accuracy}
        # tf.summary.scalar('accuracy', accuracy[1])
        # return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
