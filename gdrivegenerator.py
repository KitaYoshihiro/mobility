import numpy as np
from numpy.random import randint
from peakmodel import PeakModel
from random import shuffle
from random import random
from mschromnet_utils import BBoxUtility
import pickle

class GdriveGenerator(object):
    def __init__(self, bbox_util, batch_size, train_data, validate_data, log_intensity=False):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_data = train_data
        self.validate_data = validate_data
        self.log_intensity = log_intensity

    def generate(self, train=True, autoencoder=False):
        """
        batchサイズ分のデータを作ってyieldし続けるgenerator
        """
        while True:
            if train:
                x = np.random.permutation(len(self.train_data[0][0]))
                locations_data = self.train_data[1][x]
                input_data = self.train_data[0][0][x]
                output_data = self.train_data[0][1][x]
            else:
                x = np.random.permutation(len(self.validate_data[0][0]))
                locations_data = self.validate_data[1][x]
                input_data = self.validate_data[0][0][x]
                output_data = self.validate_data[0][1][x]

            inputs = []
            targets = []
            locations = []
            input_data_length = len(input_data[0])
            for i in np.arange(len(input_data)):
                index = randint(1024)
                indata = input_data[i][index:index+1024]
                outdata = output_data[i][index:index+1024]
                # 切り出しインデクスを normalizeした位置表現にする（切り出し前の全長における相対位置:0～1）
                trim_start_norm = (index + 0.5) / input_data_length
                trim_end_norm = (index + 1024 - 0.5) / input_data_length
                # 切り出し後の相対位置表現に変換
                witdh_norm = 1024 / input_data_length # これは0.5に決まっいてる！！
                peak_norm = locations_data[i]
                peak_norm[:, :2] -= trim_start_norm
                peak_norm[:, :2] /= witdh_norm
                left = ((peak_norm[:, 0] * 3 + peak_norm[:, 1]) / 4) >= 0
                right = ((peak_norm[:, 0] + peak_norm[:, 1] * 3) / 4) <= 1
                peak_norm_mask = left * right
                peak_norm = peak_norm[peak_norm_mask, :]

                if(random()<0.5):
                    indata = indata[::-1]
                    outdata = outdata[::-1]
                    peak_norm[:,:2] = 1 - peak_norm[:,:2]
                    peak_norm[:,:2] = peak_norm[:,:2][:,::-1]
                peak_norm = self.bbox_util.assign_boxes(peak_norm) # ここで位置情報をpriorbox表現に変換している
                locations.append(peak_norm)
                # indata = indata / np.max(indata)
                inputs.append(indata)
                # outdata = outdata / np.max(outdata)
                targets.append(outdata)
                if len(targets) == self.batch_size:
                    tmp_locations = np.array(locations, dtype='float32')
                    tmp_inp = np.array(inputs, dtype='float32')
                    tmp_targets = np.array(targets, dtype='float32')
                    locations = []
                    inputs = []
                    targets = []
                    epsilon = 0.0000001 # 10e-7
                    if self.log_intensity:
                        tmp_inp = np.log10(tmp_inp + epsilon) / 7 + 1
                        tmp_targets = np.log10(tmp_targets + epsilon) / 7 + 1
                    # yield tmp_inp, tmp_targets, tmp_locations
                    if autoencoder:
                        yield tmp_inp, tmp_targets
                    else:
                        yield tmp_inp, tmp_locations

if __name__ == '__main__':
    with open('sampledata.pkl', mode='rb') as f:
        tr = pickle.load(f)
    #chroms = np.array(tr[0])
    #locations = np.array(tr[1])
    #print(chroms.shape)
    with open('mschrom_unet_priors.pkl', mode='rb') as f:
        priors = pickle.load(f)

    bb_util = BBoxUtility(2, priors)
    gen = GdriveGenerator(bb_util, batch_size=5, train_data=tr, validate_data=tr)
    g = gen.generate(train=True)
    a = next(g)
    # a = np.array(next(g))
    print(a.shape)
