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
from cnn3d.dataset import grasping_9dir
from cnn3d.utils.model_manip import reduce_model
from keras.models import load_model
import cnn3d.utils.common as common

# color for text output
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class TrainingGrasping():
    def __init__(self, model_name, resolution):
        self._model_name = model_name
        self.icra_dataset_train_fname = params.dataset['train_fname']
        self.icra_dataset_test_fname = params.dataset['test_fname']

        self.batch_size = params.train['batch_size']
        self.nb_epoch = params.train['nb_epoch']

        self.ignore_side_grasping = params.dataset['ignore_side_grasping']
        self.centered = params.dataset['centered']

        # check if there is learned model
        self.tmp_train_dir = params.train['temp_dir']
        if not os.path.exists(self.tmp_train_dir):
            os.makedirs(self.tmp_train_dir)

        self.shape_competion = params.train['shape_completion']

        self.model_fname = self.tmp_train_dir +'grasping_' + 'epoch' + str(self.nb_epoch) \
            + '_train' \
            + str('_noside' if self.ignore_side_grasping else '') \
            + str('_centered' if self.centered else '') \
            + str('_sc' if self.shape_competion else '') \
            + '_' + model_name + '.h5'
        self._resolution = resolution

    def shape_complete(self, X, binary_mode=True):
        # [-1, 1] to [0, 1]
        X += 1.0
        X /= 2.0
        X = self.model_sc.predict(X)
        if binary_mode:
            X = np.round(X)
        # [0, 1] to [-1, 1]
        X *= 2.0
        X -= 1.0

        return X

    def train(self):
        # skip training if there is a trained model
        if os.path.isfile(self.model_fname):
            print('the model\n' + self.model_fname + '\nis existing')
            print('skip this training')
            return

        X_train, y_train, X_test, y_test, input_shape, nb_classes = grasping_9dir.load(self.icra_dataset_train_fname, self.icra_dataset_test_fname, resolution=self._resolution)

        # assert nb_classes == params.dataset['numof_class']

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # run shape completion
        if self.shape_competion:
            print('loading learned shape completion model')
            self.model_sc = load_model(params.train['sc_model_fname'])
            print('converting partial grids to completed grids ')

            X_train = self.shape_complete(X_train)
            X_test = self.shape_complete(X_test)

        # ip.embed()
        # import cnn3d.viz as viz
        # viz.show_grid(X_train[3].reshape(X_train[3].shape[1:])>0.5).show()
        # viz.show_grid(X_train_[3].reshape(X_train_[3].shape[1:])).show()

        # for 2D CNN
        X_train2D = X_train.mean(axis=4)
        X_test2D = X_test.mean(axis=4)

        if self._model_name == 'fcn':
            model = models.FCN_simple(input_shape, nb_classes)
        elif self._model_name == 'cnn':
            model = models.CNN2D_voxnet(input_shape[:-1], nb_classes)
        elif self._model_name == 'cnn3d':
            model = models.TDCNN_voxnet(input_shape, nb_classes)
        elif self._model_name == 'cnn3d_independent':
            model = models.cnn3d_independent_grasps(input_shape, nb_classes)
        elif self._model_name == 'cnn3d_side':
            model = models.cnn3d_side(input_shape, nb_classes)
        elif self._model_name == 'cnn3d_16':
            model = models.TDCNN_voxnet_16(input_shape, nb_classes)
        elif self._model_name == 'cnn3d_8':
            model = models.TDCNN_voxnet_8(input_shape, nb_classes)

        if not isinstance(model, list):
            model.summary()

        # # TEMP
        # if self._model_name == 'cnn':
        #     grasping_score, wi = self._grasping_score(model, X_test2D, y_test)
        # elif self._model_name == 'cnn3d_independent':
        #     grasping_score, wi = self._grasping_score_independent(model, X_test, y_test, verbose=True)
        # elif self._model_name == 'cnn3d_side' or self._model_name == 'fcn':
        #     grasping_score, wi = self._grasping_score_side(model, X_test, y_test, verbose=True)
        # else:
        #     grasping_score, wi = self._grasping_score(model, X_test, y_test)
        # print('Grasping score: ', grasping_score)


        if self._model_name == 'cnn':
            history, score = self._train(model, X_train2D, y_train, X_test2D, y_test, self.batch_size, self.nb_epoch)
        else:
            history, score = self._train(model, X_train, y_train, X_test, y_test, self.batch_size, self.nb_epoch)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # # TEMP
        # return

        # ip.embed()

        if self._model_name == 'cnn':
            grasping_score, wi = self._grasping_score(model, X_test2D, y_test)
        elif self._model_name == 'cnn3d_independent':
            grasping_score, wi = self._grasping_score_independent(model, X_test, y_test, verbose=True)
        elif self._model_name == 'cnn3d_side' or self._model_name == 'fcn':
            grasping_score, wi = self._grasping_score_side(model, X_test, y_test, verbose=True)
        else:
            grasping_score, wi = self._grasping_score(model, X_test, y_test)
        print('Grasping score: ', grasping_score)

        if False:
            from voxnet import viz
            viz.viz_filters_keras(model)

        # ip.embed()

        if params.train['save_model']:
            if isinstance(model, list):
                for i, m in enumerate(model):
                    m.save(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) + '_train' \
                        + str('_noside' if self.ignore_side_grasping else '') \
                        + str('_centered' if self.centered else '') \
                        + str('_sc' if self.shape_competion else '') \
                        # + str('_res') + str(self._resolution) \
                        + '_' + self._model_name + str(str(i) if i>0 else '') + '.h5')
                    np.savez(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) + '_train' \
                        + str('_noside' if self.ignore_side_grasping else '') \
                        + str('_centered' if self.centered else '') \
                        + str('_sc' if self.shape_competion else '') \
                        # + str('_res') + str(self._resolution) \
                        + '_' + self._model_name + str(str(i) if i>0 else '') + '.npz', 
                        score=score, grasping_score=grasping_score)
            else:
                # save original model
                # model.save(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) + '_train' \
                #     + str('_noside' if self.ignore_side_grasping else '') \
                #     + str('_centered' if self.centered else '') \
                #     + str('_sc' if self.shape_competion else '') \
                #     # + str('_res') + str(self._resolution) \
                #     + '_' + self._model_name + '_original' + '.h5')
                model.save('/home/lou00015/cnn3d/scripts/trained_models/grasping.h5')

                # model = reduce_model(model)
                #
                # # save original model (for faster loading and to avoid compiling)
                # model.save(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) + '_train' \
                #     + str('_noside' if self.ignore_side_grasping else '') \
                #     + str('_centered' if self.centered else '') \
                #     + str('_sc' if self.shape_competion else '') \
                #     # + str('_res') + str(self._resolution) \
                #     + '_' + self._model_name + '.h5')
                np.savez(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) + '_train' \
                    + str('_noside' if self.ignore_side_grasping else '') \
                    + str('_centered' if self.centered else '') \
                    + str('_sc' if self.shape_competion else '') \
                    # + str('_res') + str(self._resolution) \
                    + '_' + self._model_name + '.npz', 
                    score=score, grasping_score=grasping_score)

        if params.train['save_figs'] and not isinstance(model, list):
            fig_acc, fig_loss = self._plot_acc_loss(history)
            
            fig_acc.savefig(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) \
                + '_train' + '_acc' \
                + str('_noside' if self.ignore_side_grasping else '') \
                + str('_centered' if self.centered else '') \
                + str('_sc' if self.shape_competion else '') \
                # + str('_res') + str(self._resolution) \
                + '_' + self._model_name + '.pdf')

            fig_loss.savefig(self.tmp_train_dir + 'grasping_' + 'epoch' + str(self.nb_epoch) \
                + '_train' + '_loss' \
                + str('_noside' if self.ignore_side_grasping else '') \
                + str('_centered' if self.centered else '') \
                + str('_sc' if self.shape_competion else '') \
                # + str('_res') + str(self._resolution) \
                + '_' + self._model_name + '.pdf')

    def _train(self, model, x_train, y_train, x_test, y_test, batch_size, nb_epoch):
        if isinstance(model, list):
            history = []
            score = []
            # for i, m in enumerate(model):
            #     h = m.fit(x_train, y_train[:, i], batch_size=batch_size, epochs=nb_epoch,
            #         verbose=1, validation_split=0.1)
            #     s = m.evaluate(x_test, y_test[:, i], verbose=0)
            #     print(s)
            #     # score.append(s)
            #     history.append(h)

            # balanced training mode: 1 epoch for each model
            for _e in range(nb_epoch):
                for i, m in enumerate(model):
                    h = m.fit(x_train, y_train[:, i], batch_size=batch_size, epochs=1,
                        # verbose=1, validation_split=0.1)
                        verbose=1, validation_split=0.0)
                    s = m.evaluate(x_test, y_test[:, i], verbose=0)
                    print(s)
                    # score.append(s)
                    history.append(h)

                print('\n')

                for i, m in enumerate(model):
                    s = m.evaluate(x_test, y_test[:, i], verbose=0)
                    print(s)
                    score.append(s)

            # # simulate online learning
            # for x, y in zip(x_train, y_train):
            #     print('\n')
            #     for i, m in enumerate(model):
            #         s = m.evaluate(x_test, y_test[:, i], verbose=0)
            #         print(s)

            #     # x: 1xD where D is # of feature dimenion (32x32x32)
            #     # y: 1xL wehre L is # of labels
            #     x_ = x.reshape((1,)+x.shape)
            #     for i, l in enumerate(y[:len(model)]): # only consider # of models
            #         if l == 1.0:
            #             l_ = np.array(l).reshape(1, 1)
            #             model[i].train_on_batch(x_, l_)

            return history, score
        else:
            # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
            #         verbose=1, validation_data=(x_test, y_test))
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                                shuffle = True, verbose=1, validation_split=0.1)
            score = model.evaluate(x_test, y_test, verbose=0)
            return history, score
    
    def _grasping_score(self, model, X_test, y_test, verbose=False):
        n = X_test.shape[0]
        acc = 0.0
        wrong_idx = []
        # for i in range(n):
        #     p = model.predict(X_test[i, :, :, : :].reshape((1,)+X_test.shape[1:]), verbose=False)
        #     p = p[0]
        #     # wrist_idx, appr_idx = common.p2wrist_appr_indices(p)
        #     # y = y_test[i, :]
        #     # y_wrist, y_appr = common.p2wrist_p(y), common.p2appr_p(y)
        #
        #     # y_appr[appr_idx] == 1
        #
        #     if verbose:
        #         print(colors.ok + '+' + colors.close if y_wrist[wrist_idx] == 1 else colors.fail + '-' + colors.close, end='')
        #     index = np.where(p == max(p))
        #     if y_test[i][index[0]].any() == 1:
        #         acc = acc + 1
        #     else:
        #         wrong_idx.append(i)
        # print(acc/n)
        # return acc/n, wrong_idx
        return 0,0

    def _grasping_score_independent(self, model, X_test, y_test, verbose=False):
        assert isinstance(model, list)

        n = X_test.shape[0]
        acc = 0.0
        wrong_idx = []
        for i in range(n):
            p = []
            for m in model:
                p_ = m.predict(X_test[i, :, :, : :].reshape((1,)+X_test.shape[1:]), verbose=False)
                p.append(float(p_))
            # ip.embed()
            c = common.argnmax(p)
            y = y_test[i, :]

            if verbose:
                print(colors.ok + '+' + colors.close if y[c] == 1 else colors.fail + '-' + colors.close, end='')

            if y[c] == 1:
                acc = acc + 1
            else:
                wrong_idx.append(i)
        return acc/n, wrong_idx

    def _grasping_score_side(self, model, X_test, y_test, verbose=False):
        n = X_test.shape[0]
        # calculate approaching direction
        acc = 0.0
        wrong_idx = []
        for i in range(n):
            p = model.predict(X_test[i, :, :, : :].reshape((1,)+X_test.shape[1:]), verbose=False)
            p = p[0]
            w = common.argnmax(p[4:])
            y = y_test[i, 4:]

            if verbose:
                print(colors.ok + '+' + colors.close if y[w] == 1 else colors.fail + '-' + colors.close, end='')

            if y[w] == 1.0:
                acc = acc + 1
            else:
                wrong_idx.append(i)

        print('approaching direction accuracy: %f' % (acc/n))

        acc = 0.0
        wrong_idx = []
        # calculate wrist orientation
        for i in range(n):
            p = model.predict(X_test[i, :, :, : :].reshape((1,)+X_test.shape[1:]), verbose=False)
            p = p[0]
            w = common.argnmax(p[:4])
            y = y_test[i, :4]

            if verbose:
                print(colors.ok + '+' + colors.close if y[w] == 1 else colors.fail + '-' + colors.close, end='')

            if y[w] == 1.0:
                acc = acc + 1
            else:
                wrong_idx.append(i)

        print('wrist orientation accuracy: %f' % (acc/n))

        return acc/n, wrong_idx

    def _plot_acc_loss(self, history):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        fig_acc = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        # summarize history for loss
        fig_loss = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        return fig_acc, fig_loss

import argparse
if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--resolution', type=int, default=32)
    args = parser.parse_args()

    model_name = str(args.model)
    print('model_name: ', model_name)

    tg = TrainingGrasping(model_name, args.resolution)
    tg.train()
