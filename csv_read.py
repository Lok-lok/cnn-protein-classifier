import tensorflow as tf
import csv
import sys

def csv_read(path):
    try:
        file = open(path, 'r')
        csv_file = csv.reader(file)
        id_list = []
        label_list = [[False for i in range(31072)] for i in range(28)]
        next(csv_file)
        index = -1
        for stu in csv_file:
            index += 1
            id_list.append(stu[0])
            label = stu[1].split(' ')
            for i in label:
                label_list[int(i)][index] = True
        file.close()
        return id_list, label_list
    except Exception:
        print("Error! Please check path.")
        sys.exit(0)
        
id, label = csv_read("train.csv")