import numpy as np
import pickle
from peakmodel import PeakModel

class Generator(object):
    def __init__(self, batch_size, datapoints, dwelltime=1, min_peaknumber=1, max_peaknumber=10, peak_dynamicrange=3, min_peakwidth=8, max_peakwidth=200, spike_noise=True):
        self.batch_size = batch_size
        self.datapoints = datapoints
        self.dwelltime = dwelltime
        self.min_peaknumber = min_peaknumber
        self.max_peaknumber = max_peaknumber
        self.peak_dynamicrange = peak_dynamicrange
        self.min_peakwidth = min_peakwidth
        self.max_peakwidth = max_peakwidth
        self.spike_noise = spike_noise

    def generate(self, train=True):
        """
        batchサイズ分のデータを作ってyieldし続けるgenerator
        """
        while True:
            # if train:
            #     pass
            # else:
            #     pass
            inputs = []
            outputs = []
            normalized_positions = []
            for _ in np.arange(self.batch_size):
                _input, _output, _normal_positions = PeakModel.chrom(self.datapoints, dwelltime=self.dwelltime,
                                                  min_peaknumber=self.min_peaknumber,
                                                  max_peaknumber=self.max_peaknumber,
                                                  peak_dynamicrange=self.peak_dynamicrange,
                                                  min_peakwidth=self.min_peakwidth,
                                                  max_peakwidth=self.max_peakwidth)
                if self.spike_noise:
                    _input, _factor = PeakModel.normalize_and_spike(_input)
                else:
                    _input, _factor = PeakModel.normalize(_input)
                _output, _factor = PeakModel.normalize(_output, _factor)
                inputs.append(_input)
                outputs.append(_output)
                normalized_positions.append(_normal_positions)
            # yield np.array(inputs).reshape(-1,self.datapoints), np.array(outputs).reshape(-1,self.datapoints)
            yield np.array(inputs), np.array(outputs), np.array(normalized_positions)
            
if __name__ == '__main__':
    # gen = Generator(batch_size=12800, datapoints=2048, spike_noise=True)
    batch_size = 64000
    train_size = round(batch_size * 0.8)
    # gen = Generator(batch_size=batch_size, datapoints=2048, dwelltime=1,
    #                 min_peaknumber=1, max_peaknumber=10,
    #                 peak_dynamicrange=3, min_peakwidth=8,
    #                 max_peakwidth=200, spike_noise=False)
    # gen = Generator(batch_size=batch_size, datapoints=2048, dwelltime=1,
    #                 min_peaknumber=10, max_peaknumber=20,
    #                 peak_dynamicrange=4, min_peakwidth=4,
    #                 max_peakwidth=50, spike_noise=False)
    # gen = Generator(batch_size=batch_size, datapoints=2048, dwelltime=1,
    #                 min_peaknumber=1, max_peaknumber=40,
    #                 peak_dynamicrange=4, min_peakwidth=4,
    #                 max_peakwidth=300, spike_noise=False)
    gen = Generator(batch_size=batch_size, datapoints=2048, dwelltime=1,
                    min_peaknumber=1, max_peaknumber=40,
                    peak_dynamicrange=3, min_peakwidth=4,
                    max_peakwidth=200, spike_noise=False)
    g = gen.generate(train=True)
    generated = next(g)
    chroms = np.array(generated[0:2])
    ranges = np.array(generated[2])
    train_batch = (chroms[:, 0:train_size, :], ranges[0:train_size])
    validate_batch = (chroms[:, train_size:, :], ranges[train_size:])
    # with open('sharp_peaks.pickle', mode='wb') as f:
    #     pickle.dump(train_batch, f)
    with open('../trainsample3.pickle', mode='wb') as f:
        pickle.dump(train_batch, f)
    with open('../validatesample3.pickle', mode='wb') as f:
        pickle.dump(validate_batch, f)
