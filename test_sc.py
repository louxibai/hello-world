from __future__ import print_function
import sys, os
import IPython as ip
from cnn3d.dataset import sc
from cnn3d.dataset import grasping_9dir
from keras.models import load_model
from cnn3d import viz
from cnn3d import params
import numpy as np

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class ShapeCompletionTest():
    def __init__(self, model_fname, viz_fp=False, dataset='ycb'):
        
        self.dataset = dataset
        if self.dataset == 'ycb':
            self._train_fname = 'ycb.npz'
            # self._test_fname = 'ycb_campbells.npz'
            # self._test_fname = 'ycb_banana.npz'
            self._test_fname = 'ycb_all.npz'
            # self._test_fname = 'ycb.npz'
            self.X_train, self.y_train, self.input_shape = sc.load(self._test_fname)

            self.X_test = self.X_train
            self.y_test = self.y_train

        elif self.dataset == 'grasping':
            self.icra_dataset_train_fname = params.dataset['train_fname']
            self.icra_dataset_test_fname = params.dataset['test_fname']
            self.X_train, self.y_train, self.X_test, self.y_test, self.input_shape, self.nb_classes = grasping_9dir.load(self.icra_dataset_train_fname, self.icra_dataset_test_fname)

            # from [-1, 1] to [0, 1]
            self.X_train += 1.0
            self.X_train /= 2.0

            self.X_test += 1.0
            self.X_test /= 2.0

        else:
            print('unknown dataset')
            sys.exit(1)

        self.input_shape = self.input_shape[1:]

        self.X_test = self.X_test.astype('float32')
        self.y_test = self.y_test.astype('float32')

        self.model = load_model(model_fname)
        
    def predict(self, x):
        p = self.model.predict(x.reshape((1, 1,) + self.input_shape))
        return p.reshape(self.input_shape)

    def evaluate(self):
        N = self.X_test.shape[0]
        print(N, 'samples')

        idx = np.random.randint(N)

        x = self.X_test[idx, :, :, :, :].reshape(self.input_shape)
        if self.dataset == 'ycb':
            y_ = self.y_test[idx, :, :, :, :].reshape(self.input_shape)
        y = self.predict(x)

        th = 0.5
        viz.show_grid(x>th, title='partial', fig_scale=2.0).show()
        if self.dataset == 'ycb':
            viz.show_grid(y_>th, title='full', fig_scale=2.0).show()
        viz.show_grid(y>th, title='predicted', fig_scale=2.0).show()

        # save animation movies
        # viz.show_grid_ani(x>0.5, title='partial', fig_scale=2.0, save=True)
        # viz.show_grid_ani(y_>0.5, title='full', fig_scale=2.0, save=True)
        # viz.show_grid_ani(y>0.5, title='predicted', fig_scale=2.0, save=True)

        ip.embed()

if len(sys.argv) <= 1:
    print('python test_sc.py [h5 model_fname]')
    sys.exit(-1)

model_fname = str(sys.argv[1])
print('model_fname: ', model_fname)

dataset_name = str(sys.argv[2])
print('dataset_name: ', dataset_name)

ge = ShapeCompletionTest(model_fname, dataset=dataset_name)
acc = ge.evaluate()
