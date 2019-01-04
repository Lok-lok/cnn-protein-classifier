import tensorflow as tf
from numpy import array

import csv
import os
from time import gmtime, strftime
import json

import model
import csv_read

img_color = ['blue', 'green', 'red', 'yellow']

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(500).repeat().batch(batch_size)
    return dataset

def serving_input_receiver_fn():
    """
    input placeholder
    """
    inputs = {"x": tf.placeholder(dtype=tf.uint8)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
    
def main(argv):
    config_file_name = 'train_config.json'
    try:
        config_file = open(config_file_name, 'r')
        config_data = json.loads(config_file.read())
        config_file.close()
    except IOError:
        print("Fail to load configuration file")
        return 0
    if 'csv_file' not in config_data:
        print("csv file not available")
    if 'img_dir' not in config_data:
        print("image directory not available")
    if 'model_dir' not in config_data:
        print("model directory not available")
    if 'csv_file' not in config_data or 'img_dir' not in config_data or 'model_dir' not in config_data:
        return 0
    config_data['img_dir'] += "/" if config_data['img_dir'][-1] else ""
    
    img_id, label_list = csv_reader.csv_read(config_data['csv_file'])
    
    img_list = [[tf.image.decode_png(tf.read_file(config_data['img_dir'] + id + "_" + color + ".png"), dtype = tf.uint8, channels = 1) for color in img_color] for id in img_id[:1000]]
    for id in img_list:
        for color_img in id:
            color_img.set_shape([512, 512, 1])
    img = tf.convert_to_tensor([tf.stack([tf.reshape(color_img, [512, 512]) for color_img in id], axis=2) for id in img_list])
    
    label = [tf.convert_to_tensor(i[:1000]) for i in label_list]
    
    classifier = [tf.estimator.Estimator(model_fn = model.model_fn) for i in range(28)]
    
    for i in range(len(classifier)):
        classifier[i].train(input_fn = lambda:train_input_fn(img, label[i], 10), steps = 100)
        classifier[i].export_saved_model(export_dir_base=config_data['model_dir'], serving_input_receiver_fn=serving_input_receiver_fn)
        
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
    