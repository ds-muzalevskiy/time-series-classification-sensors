from keras.callbacks import EarlyStopping 

class CallBack(EarlyStopping):
    def __init__(self, threshold, min_epochs, **kwargs):
        super(CallBack, self).__init__(**kwargs)
        self.threshold = threshold 
        self.min_epochs = min_epochs 

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping on metric `%s` '
                'It is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        if (epoch >= self.min_epochs) | (current <= self.threshold):
            self.stopped_epoch = epoch
            self.model.stop_training = True