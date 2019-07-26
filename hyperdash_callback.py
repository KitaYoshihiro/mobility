from keras.callbacks import Callback

class Hyperdash(Callback):
    def __init__(self, entries, exp):
        super(Hyperdash, self).__init__()
        self.entries = entries
        self.exp = exp
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        for entrie in self.entries:
            log = logs.get(entrie)
            if log is not None:
                self.exp.metric(entrie, log)
