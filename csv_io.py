import csv
import sys

def csv_read(path):
    try:
        file = open(path, 'r')
        csv_file = csv.reader(file)
        id_list = []
        label_list = []
        next(csv_file)
        index = -1
        for stu in csv_file:
            index += 1
            id_list.append(stu[0])
            label = stu[1].split(' ')
            label_list.append([False for i in range(28)])
            for i in label:
                label_list[index][int(i)] = True
        file.close()
        return id_list, label_list
    except IOError:
        print("Error! Please check path for csv.")
        sys.exit(0)

def csv_writer(path, img, label):
    for i in range(len(label)):
        label[i] = " ".join(str(x) for x in label[i])
    csvfile = open(path, 'w', newline = "") 
    writer = csv.writer(csvfile)
    writer.writerow(["Id","Predicted"])
    length = len(img)
    for i in range(length):
        writer.writerow([img[i], label[i]])
    csvfile.close()


