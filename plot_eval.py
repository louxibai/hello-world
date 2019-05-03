# from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import IPython as ip

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.style.use('seaborn-paper')

# avoid to use Type 3 font which violates IEEE PaperPlaza
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plt.style.use('ggplot')
# plt.style.use('default')
# plt.style.use('seaborn-whitegrid')
# plt.style.use('presentation')

# fmt_list = ['-', '--', '-.']
fmt_list = ['-', '-', '-']

# CHANGE THIS LIST TO DISABLE A SEPECIFIC METHOD
ignore_list = ['fcn']
# ignore_list = []

def plot_error_bar(x, m, s, gms, title='', enable_legend=False, xlabel='', ylabel=''):
    fig, ax = plt.subplots(figsize=(4, 3))

    for i, gm in enumerate(gms):
        if gm[1] in ignore_list:
            continue
        # ax.errorbar(x, m[:, i], yerr=s[:, i], label=gm[1].upper(), fmt=fmt_list[i], linewidth=2)
        ax.plot(x, m[:, i], label=gm[1].upper())
        ax.fill_between(x, m[:, i]-s[:, i], m[:, i]+s[:, i], alpha=0.4)

    if enable_legend:
        ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.15) # for xlabel space
    return fig

from matplotlib import cm

def plot_3d(x, y, m, title='', xlabel='', ylabel=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # for color values refer to https://matplotlib.org/users/dflt_style_changes.html
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(m.shape[2]):
        Z = m[:, :, i]
        X = np.empty_like(Z)
        Y = np.empty_like(Z)
        for j in range(X.shape[1]):
            X[:, j] = x

        for k in range(Y.shape[0]):
            Y[k, :] = y

        # ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color=color_list[i], alpha=0.5, edgecolors=color_list[i])
        

    # ax.axis('equal')
    ax.set_title(title)
    ax.set_zlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('accuracy')
    ax.set_xticks(x)
    ax.set_yticks(y)
    return fig

def generate_table(m, s, name):
    str_names = ''
    str_acc = name + ' accuracy (\%) '

    for i, gm in enumerate(gms):
        str_names += '& \\{} '.format(gm[1].upper())
        str_acc += '& ${0:.2f}\pm{1:.2f}$ '.format(m[0, i]*100, s[0, i]*100)

    str_names += '\\\\'
    str_acc += '\\\\'
    return str_names + '\n' + str_acc

def plot_eval_results(acc_array, num_trials, range_nv, range_os, gms, num_interests=1, save=False):
    # title_list = ['Wrist orientation', 'Grasping direction']
    title_list = [r'\textbf{Wrist orientation} $\hat{\omega}$', r'\textbf{Grasping direction} $\hat{\delta}$']
    interest_list = ['wrist', 'appr']
    for i in range(num_interests):
        # plot acc vs noise level
        x = range_nv
        if num_interests == 1:
            y = acc_array[:, 0, :, :]
        else:
            y = acc_array[i, :, 0, :, :]
        m = np.mean(y, axis=2)
        s = np.std(y, axis=2)
        f = plot_error_bar(x, m, s, gms, 
            title=title_list[i], 
            xlabel=r'\textbf{Number of noise voxels}', ylabel=r'\textbf{Accuracy}')
        f.show()
        if save:
            f.savefig(interest_list[i]+'_nv.pdf', format='pdf', bbox_inches='tight')

        # plot acc vs occlusion level
        x = range_os
        if num_interests == 1:
            y = acc_array[0, :, :, :]
        else:
            y = acc_array[i, 0, :, :, :]
        m = np.mean(y, axis=2)
        s = np.std(y, axis=2)

        # plot_error_bar(x[:-2], m[:-2], s[:-2], gms, 
        f = plot_error_bar(x[:], m[:], s[:], gms, 
            title=title_list[i], 
            enable_legend=True if i == 0 else False, xlabel=r'\textbf{Number of occluded voxel planes}', ylabel=r'\textbf{Accuracy}')
        f.show()
        if save:
            f.savefig(interest_list[i]+'_ov.pdf', format='pdf', bbox_inches='tight')

        # # plot acc vs both noise and occlusion level (3D plot)
        # x = range_nv
        # y = range_os
        # if num_interests == 1:
        #     m = np.mean(acc_array, axis=3)
        # else:
        #     m = np.mean(acc_array[i], axis=3)
        # plot_3d(x, y, m, xlabel='Number of noise voxels', ylabel='Number of occluded voxel planes', title=title_list[i]).show()

        # generate tables
        print(generate_table(m, s, interest_list[i]))


# d = np.load('eval_grasping_1496631000.34.npz')
# d = np.load('eval_grasping_1496684052.51.npz')
# d = np.load('eval_grasping_1496711275.29.npz')
# d = np.load('eval_grasping_1496793016.78.npz')

# wrist & appr
# d = np.load('eval_grasping_1496935032.77.npz')
# d = np.load('eval_grasping_1496942069.3.npz')
# d = np.load('eval_grasping_1497001018.1.npz')
# d = np.load('eval_grasping_1497360808.86.npz')

# with banana (not bat)
# d = np.load('eval_grasping_1497538512.51.npz')
# with fcn and random as well
# d = np.load('eval_grasping_1497571857.33.npz') # very basic eval
# d = np.load('eval_grasping_1497635401.17.npz')
d = np.load('eval_grasping_1497801952.25.npz')


acc_array = d['acc_array']
try:
    num_interests = d['num_interests']
except KeyError:
    print('num_interests is not available, force to one')
    num_interests = 1
num_trials = d['num_trials']
range_nv = d['range_nv']
range_os = d['range_os']
gms = d['gms']

plot_eval_results(acc_array, num_trials, range_nv, range_os, gms, num_interests=num_interests)
ip.embed()

