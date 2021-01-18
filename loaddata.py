import numpy as np
from PIL import Image
import random
import os
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold, KFold

image_path = "image/"
image_labels_path = "image_labels/"
imagefile = []
labels = []
SIZE = 300*300
class2num = {
    'Pale-white': 0,
    'reddish': 1,
    'red': 2,
    'dull-red': 3,
    'cyanoze': 4,
    'bo-white': 5,
    'ni-white': 6,
    'bo-yellow': 7,
    'ni-yellow': 8,
}


def data2txt():

    for filename in os.listdir(image_path):
        file = image_path + filename
        n1 = file.split('.')[0]
        l1 = n1.split('/')[-1]
        # print(n1)

        with open('image_dir.txt', 'a') as fl:
            fl.writelines(image_path+l1+'.jpg'+'\n')
        with open('label_dir.txt', 'a') as fl:
            fl.writelines(image_labels_path+l1+'.txt'+'\n')

        imagefile.append(image_path+l1+'.jpg')
        labels.append(image_labels_path+l1+'.txt')
        # with open(image_labels_path + l1 +'.txt', 'w') as fl:
        #     s = l[0] + ',' + l[1]
        #     fl.writelines(s)
# 定义对应维数和各维长度的数组


def getRandomIndex(n, x): # 索引范围为[0, n), 随机选x个不重复
    index = random.sample(range(n), x)
    return index


def load_data():
    train_input_images = []
    train_input_labels = []
    test_input_images = []
    test_input_labels = []

    image_dirs = []
    label_dirs = []
    with open('image_dir.txt', 'r') as f:
        for line in f.readlines():
            image_dirs.append(line.splitlines())
    with open('label_dir.txt', 'r') as f:
        for line in f.readlines():
            label_dirs.append(line.splitlines())

    np.random.shuffle(image_dirs)

    test_index = np.array(getRandomIndex(len(image_dirs), 180))  # 再讲test_index从总的index中减去就得到了train_index
    train_index = np.delete(np.arange(len(image_dirs)), test_index)

    train_image_dirs = []
    train_label_dirs = []
    test_image_dirs = []
    test_label_dirs = []

    for i in train_index:
        train_image_dirs.append(image_dirs[i])
        train_label_dirs.append(label_dirs[i])

    for i in test_index:
        test_image_dirs.append(image_dirs[i])
        test_label_dirs.append(label_dirs[i])

    for dir in train_image_dirs:
        # print(dir[0])
        img = Image.open(dir[0])
        img = np.array(img)
        train_input_images.append(img)

        label_path = 'image_labels/' + dir[0].split('/')[1].split('.')[0] + '.txt'

        # print('{0},  {1}'.format(dir[0], label_path))

        file = open(label_path, 'r')
        text = []
        l = []
        line = file.readline()
        text.append(line)
        for t in text:
            l1 = t.split(',')[0]
            # print(l1)
            l.append(class2num[l1])
            l2 = t.split(',')[1].split('\n')[0].rstrip()
            l.append(class2num[l2])
        l = np.array(l)
        # print(l.shape)
        train_input_labels.append(l)


    # for dir in train_label_dirs:
    #     l = []
    #     # print(dir[0])
    #     file = open(dir[0], 'r')
    #     text = []
    #     line = file.readline()
    #     text.append(line)
    #     for t in text:
    #         l1 = t.split(',')[0]
    #         # print(l1)
    #         l.append(class2num[l1])
    #         l2 = t.split(',')[1].split('\n')[0].rstrip()
    #         l.append(class2num[l2])
    #     l = np.array(l)
    #     # print(l.shape)
    #     train_input_labels.append(l)
    train_input_labels = np.array(train_input_labels)

    for dir in test_image_dirs:
        # print(dir[0])
        img = Image.open(dir[0])
        img = np.array(img)
        test_input_images.append(img)

        label_path = 'image_labels/' + dir[0].split('/')[1].split('.')[0] + '.txt'

        # print('{0},  {1}'.format(dir[0], label_path))

        file = open(label_path, 'r')
        text = []
        l = []
        line = file.readline()
        text.append(line)
        for t in text:
            l1 = t.split(',')[0]
            # print(l1)
            l.append(class2num[l1])
            l2 = t.split(',')[1].split('\n')[0].rstrip()
            l.append(class2num[l2])
        l = np.array(l)
        # print(l.shape)
        test_input_labels.append(l)

    # for dir in test_label_dirs:
    #     l = []
    #     # print(dir[0])
    #     file = open(dir[0], 'r')
    #     text = []
    #     line = file.readline()
    #     text.append(line)
    #     for t in text:
    #         l1 = t.split(',')[0]
    #         # print(l1)
    #         l.append(class2num[l1])
    #         l2 = t.split(',')[1].split('\n')[0].rstrip()
    #         l.append(class2num[l2])
    #     l = np.array(l)
    #     # print(l.shape)
    #     test_input_labels.append(l)
    test_input_labels = np.array(test_input_labels)

    # encoder = LabelBinarizer()

    train_input_images = np.array(train_input_images).astype('float64') / 255
    test_input_images = np.array(test_input_images).astype('float64') / 255

    train_label1 = train_input_labels[:, 0]
    train_label2 = train_input_labels[:, 1]

    train_label1 = np_utils.to_categorical(train_label1)
    train_label2 = np_utils.to_categorical(train_label2)[:,5:9]

    test_label1 = test_input_labels[:, 0]
    test_label2 = test_input_labels[:, 1]

    test_label1 = np_utils.to_categorical(test_label1)
    test_label2 = np_utils.to_categorical(test_label2)[:, 5:9]

    return train_input_images, train_label1, train_label2, test_input_images, test_label1, test_label2


def load_data2():
    train_input_images = []
    train_input_labels = []
    image_dirs = []
    label_dirs = []
    with open('image_dir.txt', 'r') as f:
        for line in f.readlines():
            image_dirs.append(line.splitlines())
    with open('label_dir.txt', 'r') as f:
        for line in f.readlines():
            label_dirs.append(line.splitlines())
    np.random.shuffle(image_dirs)

    for dir in image_dirs:
        # print(dir[0])
        img = Image.open(dir[0])
        img = np.array(img)
        train_input_images.append(img)
        label_path = 'image_labels/' + dir[0].split('/')[1].split('.')[0] + '.txt'

        print('{0},  {1}'.format(dir[0], label_path))

        file = open(label_path, 'r')
        text = []
        l = []
        line = file.readline()
        text.append(line)
        for t in text:
            l1 = t.split(',')[0]
            # print(l1)
            l.append(class2num[l1])
            l2 = t.split(',')[1].split('\n')[0].rstrip()
            l.append(class2num[l2])
        l = np.array(l)
        # print(l.shape)
        train_input_labels.append(l)

    # for dir in label_dirs:
    #     l = []
    #     # print(dir[0])
    #     file = open(dir[0], 'r')
    #     text = []
    #     line = file.readline()
    #     text.append(line)
    #     for t in text:
    #         l1 = t.split(',')[0]
    #         # print(l1)
    #         l.append(class2num[l1])
    #         l2 = t.split(',')[1].split('\n')[0].rstrip()
    #         l.append(class2num[l2])
    #     l = np.array(l)
    #     # print(l.shape)
    #     train_input_labels.append(l)

    train_input_labels = np.array(train_input_labels)
    train_input_images = np.array(train_input_images).astype('float64') / 255

    train_label1 = train_input_labels[:, 0]
    train_label2 = train_input_labels[:, 1]

    label1 = np_utils.to_categorical(train_label1)
    label2 = np_utils.to_categorical(train_label2)[:, 5:9]  # 0 0 0 0 0 0 0 0 0 0 取后四位

    return train_input_images, train_label1, train_label2, train_input_labels, label1, label2


def load1():
    x, y1, y2, tx, ty1, ty2 = load_data()
    print(np.shape(x))
    print('y1shape:{0},  y2shape:{1}'.format(np.shape(y1), np.shape(y2)))
    print('ty1shape:{0},  ty2shape:{1}'.format(np.shape(ty1), np.shape(ty2)))
    # print(x[0].shape)
    # import pylab
    # im = x[0]
    # pylab.imshow(im)
    # pylab.show()
    #
    # tim = tx[0]
    # pylab.imshow(tim)
    # pylab.show()
    return x, y1, y2, tx, ty1, ty2

def load2():
    x, y1, y2, y, l1, l2 = load_data2()
    print('y1shape:{0},  y2shape:{1}'.format(np.shape(y1), np.shape(y2)))
    print('l1shape:{0},  l2shape:{1}'.format(np.shape(l1), np.shape(l2)))
    # cross validation
    # n-fold=5
    # skf = StratifiedKFold(n_splits=5)
    # for cnt, (train, test) in enumerate(skf.split(x, y1)):
    #     # 注意,如何取数据!当然若是df型,df.iloc[train]取值
    #     train_data = x[train, :, :]
    #     test_data = x[test, :, :]
    #     train_labels = y
    #     y_train_1 = l1[train]
    #     y_train_2 = l2[train]
    #     y_test_1 = l1[test]
    #     y_test_2 = l2[test]
    return x, y1, y2, y, l1, l2


if __name__ == '__main__':
    # data2txt()
    load2()
