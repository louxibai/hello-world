from __future__ import print_function
# import os
import numpy as np
from cnn3d import params, models
# from cnn3d.dataset import grasping_9dir


def load_train_data():
    f1 = np.load('/home/lou00015/cnn3d/dataset/cf/input_shape.npz')
    f2 = np.load('/home/lou00015/cnn3d/dataset/cf/input_pose.npz')
    x1_train = f1['g']
    y_train = f1['gt']
    x2_train = f2['g']
    x1_train = x1_train.reshape(x1_train.shape[0], 1, 32, 32, 32)
    x2_train = x2_train.reshape(x1_train.shape[0], 1, 4, 3)
    x1_train = x1_train.astype('float32')
    x2_train = x2_train.astype('float32')
    y_train = y_train.astype('float32')
    input_shape = (1, 32, 32, 32)
    input_pose = (1,4,3)
    return x1_train, x2_train, y_train, input_shape, input_pose

def load_test_data():
    f1 = np.load('/home/lou00015/cnn3d/dataset/cf/input_shape_test.npz')
    f2 = np.load('/home/lou00015/cnn3d/dataset/cf/input_pose_test.npz')
    x1_train = f1['g']
    y_train = f1['gt']
    x2_train = f2['g']
    x1_train = x1_train.reshape(x1_train.shape[0], 1, 32, 32, 32)
    x2_train = x2_train.reshape(x1_train.shape[0], 1, 4, 3)
    x1_train = x1_train.astype('float32')
    x2_train = x2_train.astype('float32')
    y_train = y_train.astype('float32')
    input_shape = (1, 32, 32, 32)
    input_pose = (1,4,3)
    return x1_train, x2_train, y_train, input_shape, input_pose


class TrainingGrasping():
    def __init__(self):
        self.batch_size = params.train['batch_size']
        self.nb_epoch = params.train['nb_epoch']

    def train(self):
        x1_train, x2_train, y_train, input_shape, input_pose = load_train_data()
        x1_test, x2_test, y_test, input_shape, input_pose = load_test_data()
        print('x1_train shape:', x1_train.shape)
        print('x2_train shape:', x2_train.shape)
        print(x1_train.shape[0], 'train samples')
        # print(X_test.shape[0], 'test samples')

        model = models.FCN_simple(input_shape, input_pose)
        model.summary()
        history, score = self._train(model, [x1_train, x2_train], y_train, [x1_test, x2_test], y_test, self.batch_size, self.nb_epoch)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        if params.train['save_model']:
            model.save('/home/lou00015/cnn3d/scripts/trained_models/cf_model.h5')

    def _train(self, model, x_train, y_train, x_test, y_test, batch_size, nb_epoch):
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                            shuffle = True, verbose=1, validation_split=0.1)
        score = model.evaluate(x_test, y_test, verbose=0)
        return history, score


if __name__ == '__main__':
    tg = TrainingGrasping()
    tg.train()


