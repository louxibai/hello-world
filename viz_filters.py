
'''
visualize convolution filters
3D voxel filters are visualized with three slided 2D filters in x, y, zeros
'''

import numpy as np
import matplotlib.pyplot as plt
import keras, sys

def viz_filters(model, layer_name = 'convolution3d_1', cm = plt.cm.gray):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    l = layer_dict[layer_name]
    w, b = l.get_weights()
    n = w.shape[0] # num of filters
    
    fx_sz, fy_sz, fz_sz = w.shape[2:]
    fcx, fcy, fcz = (fx_sz-1)/2, (fy_sz-1)/2, (fz_sz-1)/2 # center indices

    plt.figure(figsize=(16,2))

    for i in range(n):
        f_ = w[i, 0, :, :, :]
        # slices
        f_x = f_[fcx, :, :]
        f_y = f_[:, fcy, :]
        f_z = f_[:, :, fcz]

        plt.subplot(3, n, n*0 + i + 1)
        plt.imshow(f_x.transpose(), cmap=cm, interpolation='nearest')
        plt.axis('off')

        plt.subplot(3, n, n*1 + i + 1)
        plt.imshow(f_y.transpose(), cmap=cm, interpolation='nearest')
        plt.axis('off')

        plt.subplot(3, n, n*2 + i + 1)
        plt.imshow(f_z.transpose(), cmap=cm, interpolation='nearest')
        plt.axis('off')
    # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.0)
    plt.show()


if len(sys.argv) < 2:
    print('python viz_filters.py [model_fname]')
    sys.exit(1)

model_fname = str(sys.argv[1])
print('model_fname: ', model_fname)

model = keras.models.load_model(model_fname)
viz_filters(model)

