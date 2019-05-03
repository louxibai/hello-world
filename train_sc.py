'''
train 3d cnn model
'''
from __future__ import print_function

import numpy as np
import sys, os
import IPython as ip
import matplotlib
import matplotlib.pyplot as plt

from cnn3d import params, models
from cnn3d.dataset import sc
from cnn3d.utils.model_manip import reduce_model

# color for text output
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class Train():
    def __init__(self, model_name, train_fname='ycb.npz'):
        self._model_name = model_name
        self._train_fname = train_fname

        self.batch_size = params.train['batch_size']
        # self.batch_size = 4 # sc_cu
        # self.batch_size = 8 # sc_cu
        # self.batch_size = 16
        # self.batch_size = 32 # sc
        self.batch_size = 64 # sc
        self.nb_epoch = params.train['nb_epoch']
        # self.nb_epoch = 4
        # self.nb_epoch = 10
        # self.nb_epoch = 20
        # self.nb_epoch = 40
        # self.nb_epoch = 300
        self.nb_epoch = 200

        # check if there is learned model
        self.tmp_train_dir = params.train['temp_dir']
        if not os.path.exists(self.tmp_train_dir):
            os.makedirs(self.tmp_train_dir)

        self.model_fname = self.tmp_train_dir +'sc_' + 'epoch' + str(self.nb_epoch) \
            + '_' + model_name + '.h5'

    def train(self):
        # skip training if there is a trained model
        if os.path.isfile(self.model_fname):
            print('the model\n' + self.model_fname + '\nis existing')
            print('skip this training')
            return

        print('loading traning dataset...')
        X_train, y_train, input_shape = sc.load(self._train_fname)
        # input_shape = input_shape[1:]
        print('input_shape: '); print(input_shape)

        X_train = X_train.astype('float32')
        y_train = y_train.astype('float32')

        print('loading sc model...')
        if self._model_name == 'sc':
            model = models.SC(input_shape)
        elif self._model_name == 'sc_cu':
            model = models.SC_CU(input_shape)

        history = self._train(model, X_train, y_train, self.batch_size, self.nb_epoch)

        if params.train['save_model']:
            model = reduce_model(model)
            model.save(self.tmp_train_dir + 'sc_' + 'epoch' + str(self.nb_epoch) + '_train_' \
                + self._model_name + '.h5')
        if params.train['save_figs']:
            fig_acc, fig_loss = self._plot_acc_loss(history)

            fig_acc.savefig(self.tmp_train_dir + 'sc_' + 'epoch' + str(self.nb_epoch) \
                + '_acc_' + self._model_name + '.pdf')

            fig_loss.savefig(self.tmp_train_dir + 'sc_' + 'epoch' + str(self.nb_epoch) \
                + '_loss_' + self._model_name + '.pdf')

    def _train(self, model, x_train, y_train, batch_size, nb_epoch):
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.1)
        return history
    
    def _plot_acc_loss(self, history):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        fig_acc = plt.figure()
        plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        plt.legend(['train'], loc='upper left')

        # summarize history for loss
        fig_loss = plt.figure()
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        plt.legend(['train'], loc='upper left')

        return fig_acc, fig_loss
        # return fig_loss


if len(sys.argv) <= 1:
    print('python train_sc.py [model_name] [train_fname]')
    sys.exit(1)

model_name = str(sys.argv[1])
train_fname = str(sys.argv[2])
print('model_name: ', model_name)
print('train_fname: ', train_fname)

tg = Train(model_name, train_fname=train_fname)
tg.train()
