from __future__ import print_function
import sys, os
import IPython as ip
from cnn3d.dataset import grasping_9dir
from cnn3d.baseline.pca_grasping import PCAGrasping
from cnn3d.baseline.svm_grasping import SVMGrasping
from cnn3d.baseline.rand_grasping import RANDGrasping
from cnn3d.grasping import CNN3dGrasping
from cnn3d import params
import numpy as np
from cnn3d.utils.grid_manip import rotate_grid
import cnn3d.utils.common as common
from keras.models import load_model
import time
import cnn3d.viz as viz
import matplotlib.pyplot as plt



class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

css = """
html { margin: 0 }
body {
    background:#fff;
    color:#000;
    font:75%/1.5em Helvetica, "DejaVu Sans", "Liberation sans", "Bitstream Vera Sans", sans-serif;
    position:relative;
}
/*dt { font-weight: bold; float: left; clear; left }*/
div { padding: 10px; width: 80%; margin: auto }
img { border: 1px solid #eee }
dl { margin:0 0 1.5em; }
dt { font-weight:700; }
dd { margin-left:1.5em; }
table {
    border-collapse:collapse;
    border-spacing:0;
    margin:0 0 1.5em;
    padding:0;
}
td { padding:0.333em;
    vertical-align:middle;
}
}"""

import sys
import cStringIO

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

class html_writer():
    def __init__(self, fname, imgfmt='pdf'):
        self._fname = fname
        self._imgfmt = imgfmt
        with open(self._fname, 'w') as f:
            f.write('<html><head><style>')
            f.write(css)
            f.write('</style></head>')
            f.write('<body>')

    def close(self):
        with open(self._fname, 'a') as f:
            f.write('</body></html>')

    def fig2pdf(self, fig):
        sio = cStringIO.StringIO()
        # fig.savefig(sio, format=self._imgfmt, bbox_inches='tight')
        fig.savefig(sio, format=self._imgfmt)
        return sio.getvalue().encode("base64").strip()

    def add(self, X, X_idx, p, wrist_idx, appr_idx, y_wrist=[], y_appr=[], X_orig=np.array([]), with_gt=False):

        f_gt = viz.show_vgrid(X, p_appr=y_appr, p_wrist=y_wrist)
        f_pred = viz.show_vgrid(X, p_appr=common.p2appr_p(p), p_wrist=common.p2wrist_p(p))
        if X_orig.size > 0:
            f_gt_orig = viz.show_vgrid(X_orig, p_appr=y_appr, p_wrist=y_wrist)

        with open(self._fname, 'a') as f:
            f.write('<div>')
            f.write('<table><tr><td>id:{}</td><td>'.format(X_idx))
            if X_orig.size > 0 and with_gt:
                if self._imgfmt == 'pdf':
                    f.write('<embed src="data:application/%s;base64,%s" width="360" height="360" type="application/pdf">' % (self._imgfmt, self.fig2pdf(f_gt_orig)))
                else:
                    f.write('<img src="data:image/%s;base64,%s"/>' % (self._imgfmt, self.fig2pdf(f_gt_orig)))

            if with_gt:
                if self._imgfmt == 'pdf':
                    f.write('<embed src="data:application/%s;base64,%s" width="360" height="360" type="application/pdf">' % (self._imgfmt, self.fig2pdf(f_gt)))
                else:
                    f.write('<img src="data:image/%s;base64,%s"/>' % (self._imgfmt, self.fig2pdf(f_gt)))
            # f.write('<br>')
            if self._imgfmt == 'pdf':
                f.write('<embed src="data:application/%s;base64,%s" width="360" height="360" type="application/pdf">' % (self._imgfmt, self.fig2pdf(f_pred)))
            else:
                f.write('<img src="data:image/%s;base64,%s"/>' % (self._imgfmt, self.fig2pdf(f_pred)))
            f.write('</td>')
            f.write('<td>')
            f.write('<dl><dt>Approaching directions</dt>')
            f.write('<dt>ground truth:</dt><dd>{}</dd>'.format(y_appr))
            f.write('<dt>predction:</dt><dd>{}</dd>'.format(common.p2appr_p(p)))
            f.write('<dt>chosen approaching direction index:</dt><dd><b><font color="{}">{}</font></b></dd></dl>'.format('green' if y_appr[appr_idx] else 'red', appr_idx))
            f.write('<dl><dt>Wrist orientations</dt>')
            f.write('<dt>ground truth:</dt><dd>{}</dd>'.format(y_wrist))
            f.write('<dt>predction:</dt><dd>{}</dd>'.format(common.p2wrist_p(p)))
            f.write('<dt>chosen wrist orientation index:</dt><dd><b><font color="{}">{}</font></b></dd></dl>'.format('green' if y_wrist[wrist_idx] else 'red', wrist_idx))
            f.write('</td></tr></table>')
            f.write('</div>')
        plt.close(f_gt)
        plt.close(f_pred)
        if X_orig.size > 0:
            plt.close(f_gt_orig)

class GraspEvaluation():
    def __init__(self, grasping_methods, viz_fp=False, entire=True, viz=False, side_grasp=False, noise_voxels=0, occluded_slices=0, resolution=32):
        if noise_voxels > 0 or occluded_slices > 0:
            self.test_challenging = True
        else:
            self.test_challenging = False

        if self.test_challenging:
            self.X_train, self.y_train, self.X_test_orig, self.y_test, self.input_shape, self.nb_classes, self.X_test = grasping_9dir.load(numof_noisy_voxels=noise_voxels, numof_occluded_slices=occluded_slices, resolution=resolution)
        else:
            self.X_train, self.y_train, self.X_test, self.y_test, self.input_shape, self.nb_classes = grasping_9dir.load(numof_noisy_voxels=noise_voxels, numof_occluded_slices=occluded_slices, resolution=resolution)

        self.shape_competion = params.train['shape_completion']
        if self.shape_competion:
            print('loading learned shape completion model')
            self.model_sc = load_model(params.train['sc_model_fname'])
            print('converting partial grids to completed grids ')

            # self.X_train = self.shape_complete(self.X_train)
            self.X_test = self.shape_complete(self.X_test)

        self.grasping_methods = grasping_methods
        self._viz_fp = viz_fp
        self._entire = entire
        self._viz = viz
        self._side_grasp = side_grasp

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

    def print_verbose(self, X, p, wrist_idx, appr_idx, y_wrist=[], y_appr=[]):
        print('')
        print('approaching direction')
        print('prediced:'); print(common.p2appr_p(p)); print(appr_idx)
        if len(y_appr) > 0:
            print('gt:'); print(y_appr); print(common.argnmax(y_appr))

        print('wrist orientation')
        print('prediced:'); print(common.p2wrist_p(p)); print(wrist_idx)
        if len(y_wrist) > 0:
            print('gt:'); print(y_wrist); print(common.argnmax(y_wrist))

    def evaluate(self):
        # self.X_test = self.X_test.astype('float32')
        N = self.X_test.shape[0]
        print(N, 'test samples')
        assert self.X_test.shape[0] == self.y_test.shape[0]

        if (self._viz_fp or self._viz) and self._entire:
            hw = True
        else:
            hw = False

        accuracy = []
        for gm, gm_name in self.grasping_methods:
            print(gm_name)
            correct_wrist = 0
            correct_appr = 0
            # correct_both = 0

            if hw:
                hwfn = 'viz_' + gm_name
                if self._viz_fp:
                    hwfn += '_fp'
                elif self._viz:
                    hwfn += '_all'
                hwfn += '.html'
                self._hw = html_writer(hwfn, imgfmt='pdf')

            for i in range(N) if self._entire else range(100):
                X = self.X_test[i, 0, :]
                p = gm.predict(X)
                wrist_idx, appr_idx = common.p2wrist_appr_indices(p)

                y = self.y_test[i, :]

                y_wrist, y_appr = common.p2wrist_p(y), common.p2appr_p(y)

                print('(' + (common.print_pos() if y_appr[appr_idx] == 1 else common.print_neg()) + (common.print_pos() if y_wrist[wrist_idx] == 1 else common.print_neg()) + ')', end='')

                if y_wrist[wrist_idx] == 1:
                    correct_wrist += 1

                if y_appr[appr_idx] == 1:
                    correct_appr += 1

                if self._viz:
                    # self.print_verbose(X, p, wrist_idx, appr_idx, y_wrist, y_appr)
                    if hw:
                        if self.test_challenging:
                            X_orig = self.X_test_orig[i, 0, :]
                            self._hw.add(X, i, p, wrist_idx, appr_idx, y_wrist, y_appr, X_orig, with_gt=True if gm_name == 'cnn3d' else False)
                        else:
                            self._hw.add(X, i, p, wrist_idx, appr_idx, y_wrist, y_appr)
                elif (y_wrist[wrist_idx] != 1 or y_appr[appr_idx] != 1) and self._viz_fp:
                    # self.print_verbose(X, p, wrist_idx, appr_idx, y_wrist, y_appr)
                    if hw:
                        if self.test_challenging:
                            X_orig = self.X_test_orig[i, 0, :]
                            self._hw.add(X, i, p, wrist_idx, appr_idx, y_wrist, y_appr, X_orig)
                        else:
                            self._hw.add(X, i, p, wrist_idx, appr_idx, y_wrist, y_appr)

                # if self._side_grasp:
                #     # print('up direction')
                #     g = X
                #     p = gm.predict(g)
                #     wrist_idx, appr_idx = common.p2wrist_appr_indices(p)
                #     # self.print_verbose(g, p, wrist_idx, appr_idx)

                #     # print('front direction')
                #     g = rotate_grid(X, 0, 90, 0)
                #     p = gm.predict(g)
                #     wrist_idx, appr_idx = common.p2wrist_appr_indices(p)
                #     # self.print_verbose(g, p, wrist_idx, appr_idx)
                    
                #     # print('left direction')
                #     g = rotate_grid(X, 90, 0, 90)
                #     p = gm.predict(g)
                #     wrist_idx, appr_idx = common.p2wrist_appr_indices(p)
                #     # self.print_verbose(g, p, wrist_idx, appr_idx)

                #     # print('right direction')
                #     g = rotate_grid(X, -90, 0, -90)
                #     p = gm.predict(g)
                #     wrist_idx, appr_idx = common.p2wrist_appr_indices(p)

                #     # self.print_verbose(g, p, wrist_idx, appr_idx)

            if self._entire:
                # acc = float(correct_both)/N
                acc_wrist = float(correct_wrist)/N
                acc_appr = float(correct_appr)/N
                print('')
                # print('accuracy: ', acc)
                print('accuracy (wrist): ', acc_wrist)
                print('accuracy (appr): ', acc_appr)
                accuracy.append((acc_wrist, acc_appr))
            else:
                print('')

            if hw:
                self._hw.close()
        return accuracy

import argparse
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, required=True, help='grasping method (pca, svm, cnn3d, fcn, rand)')
    parser.add_argument('--entire', type=int, default=1)
    parser.add_argument('--viz', type=int, default=0)
    parser.add_argument('--viz_fp', type=int, default=0)
    # parser.add_argument('--num_pc', type=int, default=2, help='if pca is chosen')
    # parser.add_argument('--model_fname', type=str, default='/tmp/cnn3d_train/grasping_epoch150_train_noside_cnn3d.h5', help='required if cnn3d is chosen')
    parser.add_argument('--model_fname', type=str, default='/tmp/grasping_epoch150_train_cnn3d_side.h5', help='required if cnn3d is chosen')
    parser.add_argument('--model_fname_fcn', type=str, default='/tmp/grasping_epoch1000_train_fcn.h5', help='required if cnn3d is chosen')
    # parser.add_argument('--model_fname', type=str, default='/tmp/cnn3d_train/grasping_epoch30_train_noside_cnn3d.h5', help='required if cnn3d is chosen')
    parser.add_argument('--side_grasp', type=int, default=0, help='enable to test side grasping')
    parser.add_argument('--noise_voxels', type=int, default=0, help='number of noisy voxels in testing dataset')
    parser.add_argument('--occluded_slices', type=int, default=0, help='number of occluded (x-z) slices in testing dataset')
    parser.add_argument('--model_fname_svm', type=str, default='/tmp/svm_grasp_baseline_iter1000.pkl', help='required if svm is chosen')
    # parser.add_argument('--model_fname_svm', type=str, default='/tmp/svm_grasp_baseline_iter10.pkl', help='required if svm is chosen')
    # parser.add_argument('--model_fname_svm', type=str, default='/tmp/svm_grasp_baseline_centered_iter2000.pkl', help='required if svm is chosen')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=32)
    args = parser.parse_args()

    print('grasping method %s is chosen.' % args.method)

    gms = []
    if args.method == 'pca':
        gm = PCAGrasping()
        gms.append((gm, 'pca'))
    elif args.method == 'svm':
        gm = SVMGrasping(args.model_fname_svm)
        gms.append((gm, 'svm'))
    elif args.method == 'cnn3d':
        gm = CNN3dGrasping(args.model_fname)
        gms.append((gm, 'cnn3d'))
    elif args.method == 'fcn':
        gm = CNN3dGrasping(args.model_fname_fcn)
        gms.append((gm, 'fcn'))
    elif args.method == 'rand':
        gm = RANDGrasping()
        gms.append((gm, 'rand'))
    elif args.method == 'all':
        gm = RANDGrasping()
        gms.append((gm, 'rand'))
        gm = PCAGrasping()
        gms.append((gm, 'pca'))
        gm = SVMGrasping(args.model_fname_svm)
        gms.append((gm, 'svm'))
        gm = CNN3dGrasping(args.model_fname_fcn)
        gms.append((gm, 'fcn'))
        gm = CNN3dGrasping(args.model_fname)
        gms.append((gm, 'cnn3d'))
    else:
        sys.exit('unknown grasping method: %s' % args.method)

    if not args.batch:
        # run only once
        ge = GraspEvaluation(gms, viz=args.viz, entire=args.entire, viz_fp=args.viz_fp, side_grasp=args.side_grasp, noise_voxels=args.noise_voxels, occluded_slices=args.occluded_slices, resolution=args.resolution)
        acc = ge.evaluate()
        print(acc)
    else:
        # run multiple times in a batch mode

        num_interests = 2 # approaching direction, wrist orientation

        # very quick evaluation (good for checking all methods are working)
        # num_trials = 1
        # range_nv = range(0, 1000+1, 1000)
        # range_os = range(0, 4+1, 4)

        # num_trials = 3
        # range_nv = range(0, 2000+1, 1000)
        # range_os = range(0, 8+1, 4)

        # num_trials = 10
        # range_nv = range(0, 10000+1, 2000)
        # range_os = range(0, 24+1, 8)


        # num_trials = 10
        # range_nv = range(0, 10000+1, 2000)
        # range_os = range(0, 24+1, 8)

        # num_trials = 10
        # range_nv = range(0, 10000+1, 1000)
        # range_os = range(0, 24+1, 4)

        # extensive evaluation (good for paper)
        num_trials = 30
        range_nv = range(0, 10000+1, 1000)
        range_os = range(0, 24+1, 4)

        num_methods = len(gms)
        acc_array = np.zeros((num_interests, len(range_nv), len(range_os), len(gms), num_trials), dtype=np.float32)
        for i in range(num_trials):
            for j, nv in enumerate(range_nv):
                for k, os in enumerate(range_os):
                    print('[%d/%d][%d/%d][%d/%d]' % (i, num_trials, j, len(range_nv), k, len(range_os)))
                    ge = GraspEvaluation(gms, noise_voxels=nv, occluded_slices=os)
                    acc = ge.evaluate()
                    for l in range(num_methods):
                        acc_array[0, j, k, l, i] = acc[l][0] # wrist
                        acc_array[1, j, k, l, i] = acc[l][1] # appr

        # ip.embed()

        np.savez('eval_grasping_' + str(time.time()) + '.npz', 
            acc_array=acc_array,
            num_interests=num_interests, 
            num_trials=num_trials, 
            range_nv=range_nv, 
            range_os=range_os, 
            gms=gms)





