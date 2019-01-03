import tensorflow as tf
import csv
import os
from sklearn.cross_validation import train_test_split

def csv_reader(path):

    try:
        file = open(path, 'r')
        csv_file = csv.reader(file)
        img_list = [[] for i in range(28)]
        next(csv_file)
        for stu in csv_file:
            label = stu[1].split(' ')
            for i in label:
                img_list[int(i)].append(stu[0])
        return img_list
    except Exception:
        print ("Error! Please check path.")
    finally:
        file.close()