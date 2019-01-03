import tensorflow as tf
import csv
import os
import model
import csv_reader
from time import gmtime, strftime

def main(argv):
    # load data: (train_images, train_labels), (test_images, test_labels)
    #seed = 10
    X = 
    Y = 
    train_images, train_labels, test_images, test_labels = train_test_split(X, Y, test_size=0.2)

    classifiers = [tf.estimator.Estimator(model_fn = model_fn) for i in range(28)]
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = train_images,
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = test_images,
        y = test_labels,
        num_epochs = 1,
        shuffle = False)
        
    for classifier in classifiers:
        classifier.train(input_fn = train_input_fn, steps = 10000)

if __name__ == "__main__":
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
    time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    save_path = saver.save(session, "/tmp/" + time + ".ckpt")

