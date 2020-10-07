from gensim.models.callbacks import CallbackAny2Vec

class EpochSaver(CallbackAny2Vec):
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
    def on_epoch_end(self, model):
        #output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        self.epoch += 1

