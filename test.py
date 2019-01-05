import tensorflow as tf
from numpy import array
import csv
import os
from time import gmtime, strftime
import json
import model
import csv_read

img_color = ['blue', 'green', 'red', 'yellow']
batch_size = 16

def eval_input_fn(img_id, img_dir, labels, batch_size):
    img_list = [[tf.image.decode_png(tf.read_file(img_dir + id + "_" + color + ".png"), dtype = tf.uint8, channels = 1) for color in img_color] for id in img_id]
    for id in img_list:
        for color_img in id:
            color_img.set_shape([512, 512, 1])
    img = [tf.divide(tf.cast(tf.stack([tf.reshape(color_img, [512, 512]) for color_img in id], axis=2), dtype = tf.float32), tf.convert_to_tensor(255.0)) for id in img_list]
    #img = dict(img)
    labels = tf.cast(labels, tf.int32)
    if labels is None:
        inputs = img
    else:
        inputs = (img, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset
    
def main(argv):
    config_file_name = 'test_config.json'
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
    if 'ckpt_dir' not in config_data:
        print("checkpoint directory not available")
    if 'csv_file' not in config_data or 'img_dir' not in config_data or 'model_dir' not in config_data or 'ckpt_dir' not in config_data:
        return 0
    config_data['img_dir'] += "/" if config_data['img_dir'][-1] else ""
    
    img_id, label_list = csv_read.csv_read(config_data['csv_file'])
    
    classifier = [tf.estimator.Estimator(model_fn = model.model_fn, model_dir = config_data['ckpt_dir'] + "/model_" + str(i)) for i in range(28)] 

    for i in range(len(classifier)):
        #eval_result = classifier[i].evaluate(input_fn = lambda:eval_input_fn(img_id[100:], config_data['img_dir'], label_test[i], batch_size), steps = 1)
        
        print ("==============================================================================================================================================")
        predictions = classifier[i].predict(input_fn=lambda:eval_input_fn(img_id[1:], config_data['img_dir'], None, batch_size = batch_size))
        for predict in predictions:
            index = predict[:, tf.newaxis]
            print (index[0])
        print ("==============================================================================================================================================")

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()