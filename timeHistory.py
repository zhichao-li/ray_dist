from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow


class TimeHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.batch_time = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def on_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs=None):
        self.batch_time.append(time.time() - self.batch_time_start)