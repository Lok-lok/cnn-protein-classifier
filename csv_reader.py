import tensorflow as tf
import csv
import os


def csv_reader(path):

    try:
        file = open(path, 'r')
        csv_file = csv.reader(file)
        img_list = []
        label_list = [[0 for i in range(31072)] for i in range(28)]
        next(csv_file)
        index = -1
        for stu in csv_file:
            index += 1
            img_list.append("/train/"+ stu[0] + ".png")
            label = stu[1].split(' ')
            for i in label:
                label_list[int(i)][index] = 1
        return img_list, label_list
    except Exception:
        print ("Error! Please check path.")
    finally:
        file.close()


img, lab = csv_reader("train.csv")
print (img)