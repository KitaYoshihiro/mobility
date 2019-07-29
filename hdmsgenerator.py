import numpy as np
from numpy.random import randint
from random import shuffle
from random import random
import pickle
from keras.utils import to_categorical

class HDMSGenerator(object):
    def __init__(self, train_X, train_y):
        self.batch_size = 8
        traindata = list(zip(train_X, train_y))
        traindata.sort(key=lambda x: x[1])
        self.train_X, self.train_y = zip(*traindata)
        self.offsets = [0, len(train_y[train_y == 0]), len(train_y)]

    def generate(self):
        """
        batchサイズ分のデータを作ってyieldし続けるgenerator
        """
        while True:
            ret_X = []
            ret_X2 = []
            ret_y = np.random.randint(0, 2, self.batch_size)
            for i in range(self.batch_size):
                classid = ret_y[i]
                #is_blend = np.random.random() > 0.2
                is_blend = False
                #is_blend = np.random.randint(0, 2, 1) # 二択（シングルかブレンドか　0シングル　1ブレンド）
                XIDs = np.random.randint(self.offsets[classid], self.offsets[classid+1], 2)
                if is_blend:
                    f = np.random.beta(0.5, 0.5, 1)
                    synX = self.train_X[XIDs[0]] * f + self.train_X[XIDs[1]] * (1-f)
                    ret_X.append(synX)
                else:
                    ret_X.append(self.train_X[XIDs[0]])
                ret_X2.append(self.train_X[XIDs[1]])
            yield np.array(ret_X)[:,:,:,np.newaxis], [to_categorical(ret_y, num_classes=2), np.array(ret_X2).reshape(self.batch_size, -1)]

if __name__ == '__main__':
    with open('../20190719_bin_1_ndarray.pickle', mode='rb') as f:
        X = pickle.load(f)
        X = X[0:150,4:196,]
    with open('../20190719_class_ndarray.pickle', mode='rb') as f:
        y = pickle.load(f)

    gen = HDMSGenerator(train_X=X, train_y=y)
    g = gen.generate()
    a, b = next(g)
    print(np.array(a).shape, '\n', b)
