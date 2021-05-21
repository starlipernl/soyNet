import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage import exposure
from skimage.io import imsave
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix as cm, accuracy_score as acc

data_dir = 'imageData'
option = 'plot'
matchstring = '4-228'
hog_orient = 12
hog_cellsize = (12, 12)
hog_blocksize = (2, 2)
hog_trans_sqrt = True

print('Loading Data')
# load data
x_train = np.load(os.path.join(data_dir, 'x_train_' + option + '_' + matchstring + '.npy'))
x_val = np.load(os.path.join(data_dir, 'x_test_' + option + '_' + matchstring + '.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train_' + option + '_' + matchstring + '.npy'))
y_val = np.load(os.path.join(data_dir, 'y_test_' + option + '_' + matchstring + '.npy'))
y_train[y_train == 5] = 4
y_val[y_val == 5] = 4
# crop images for memory constraints
x_train = x_train[:, 120:, :520, :]
x_val = x_val[:, 120:, :520, :]

#
# # initialize arrays
# hog_train = []
# hog_train_viz = []
#
# hog_val = []
#
# # calculate hog features and visualizations
# print('Extracting HoG features for X_trian')
# for img in x_train:
#
#     hog_feat = hog(img, orientations=hog_orient, pixels_per_cell=hog_cellsize,
#                    cells_per_block=hog_blocksize, transform_sqrt=hog_trans_sqrt,
#                    visualize=False, multichannel=True)
#     hog_train.append(hog_feat)
#     # hog_train_viz.append(hog_viz)
#
# # hog_viz_show = exposure.rescale_intensity(hog_train_viz[0], in_range=(0, 10))
# # imsave('hog_viz0_o10_pix8_blck2_normalize.png', hog_viz_show)
# print('Extracting HoG features for X_val')
# for img in x_val:
#
#     hog_feat = hog(img, orientations=hog_orient, pixels_per_cell=hog_cellsize,
#                    cells_per_block=hog_blocksize, transform_sqrt=hog_trans_sqrt,
#                    visualize=False, multichannel=True)
#     hog_val.append(hog_feat)
#
# hog_train = np.asarray(hog_train)
# hog_val = np.asarray(hog_val)
#
# np.save('hog_train_o8pix4block2_trans.npy', hog_train)
# np.save('hog_val_o8pix4block2_trans.npy', hog_val)

hog_train = np.load('hog_train_o8pix4block2_trans.npy')
hog_val = np.load('hog_val_o8pix4block2_trans.npy')

# c_grid = [0.25, 0.5, 0.75, 0.9, 1]
c_grid = [1, 0.1]

for cc in c_grid:
    print('Training SVM for C={}'.format(cc))
    clf = svm.SVC(C=cc, cache_size=1000, class_weight='balanced')
    clf.fit(hog_train, np.asarray(y_train))
    print('Running Inference SVM')
    # score = clf.score(hog_val, np.asarray(y_val))
    # print('Inference Score: ', score)
    predictions = clf.predict(hog_val)
    confusion = cm(np.asarray(y_val), predictions)
    print(confusion)
    print(acc(np.asarray(y_val), predictions))

