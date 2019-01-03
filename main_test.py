import tensorflow as tf
import csv
import os
from sklearn.cross_validation import train_test_split
import model
import csv_reader

def main(argv):
    # load data: (train_images, train_labels), (test_images, test_labels)
    #seed = 10
    X = 
    Y = 
    train_images, train_labels, test_images, test_labels = train_test_split(X, Y, test_size=0.2)

    classifiers = [tf.estimator.Estimator(model_fn = model_fn) for i in range(28)]
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = test_images,
        y = test_labels,
        num_epochs = 1,
        shuffle = False)
        
    for classifier in classifiers:
        eval_results = classifier.evaluate(input_fn = eval_input_fn)
        print(eval_results)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

