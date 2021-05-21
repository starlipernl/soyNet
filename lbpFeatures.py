import os
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv
from sklearn.metrics import confusion_matrix as cm, accuracy_score as acc
from sklearn import svm
from scipy.io import savemat, loadmat
import cv2

data_dir = 'imageData'
option = 'plot'
matchstring = '10-87'

print('Loading Data')
# load data
x_train = np.load(os.path.join(data_dir, 'x_train_' + option + '_' + matchstring + '.npy'))
x_val = np.load(os.path.join(data_dir, 'x_test_' + option + '_' + matchstring + '.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train_' + option + '_' + matchstring + '.npy'))
y_val = np.load(os.path.join(data_dir, 'y_test_' + option + '_' + matchstring + '.npy'))
y_train[y_train == 5] = 4
y_val[y_val == 5] = 4

# LBP parameters
radius = [1, 2, 4, 8]
n_points = [8 * r for r in radius]
METHOD = 'uniform'
# initialize arrays
lbp_train = []
lbp_val = []

# calculate lbp features multi-scale
print('Extracting LBP features for X_trian')
for img in x_train:
    lbp_feat = []
    # concatenate multiple scales of LBP radius and points
    for rind, r in enumerate(radius):
        img = img[:, :, [2, 1, 0]]
        img_gray = rgb2gray(img)
        lbp = local_binary_pattern(img_gray, n_points[rind], r, METHOD)
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        lbp_feat.extend(lbp_hist)
    # extract LBP for HSV hue channel
    for rind, r in enumerate(radius):
        img_hsv = rgb2hsv(img)
        lbp = local_binary_pattern(img_hsv[:, :, 0], n_points[rind], r, METHOD)
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        lbp_feat.extend(lbp_hist)

    lbp_train.append(lbp_feat)

print('Extracting LBP features for X_val')
for img in x_val:
    lbp_feat = []
    for rind, r in enumerate(radius):
        img = img[:, :, [2, 1, 0]]
        img_gray = rgb2gray(img)
        lbp = local_binary_pattern(img_gray, n_points[rind], r, METHOD)
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        lbp_feat.extend(lbp_hist)
    # extract LBP for HSV hue channel
    for rind, r in enumerate(radius):
        img_hsv = rgb2hsv(img)
        lbp = local_binary_pattern(img_hsv[:, :, 0], n_points[rind], r, METHOD)
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        lbp_feat.extend(lbp_hist)

    lbp_val.append(lbp_feat)

lbp_train = np.asarray(lbp_train)
lbp_val = np.asarray(lbp_val)

np.save('lbp_train_rad{}_npoints{}_{}_hsv_10-87.npy'.format(radius, n_points, METHOD), lbp_train)
np.save('lbp_val_rad{}_npoints{}_{}_hsv_10-87.npy'.format(radius, n_points, METHOD), lbp_val)

mrelbp_feats = loadmat('mreLBP_feats1248Samples8xr_10-87.mat')
lbp_train = mrelbp_feats['mrelbpTrain']
lbp_val = mrelbp_feats['mrelbpVal']


# c_grid = [0.25, 0.5, 0.75, 0.9, 1]
c_grid = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 5000, 10000]

predictions = []
for cc in c_grid:
    print('Training SVM for C={}'.format(cc))
    clf = svm.SVC(C=cc, cache_size=1000, class_weight='balanced')
    clf.fit(lbp_train, np.asarray(y_train)) # mrelbp_feats['mrelbpTrain']
    print('Running Inference SVM')
    # score = clf.score(hog_val, np.asarray(y_val))
    # print('Inference Score: ', score)
    predictions.append(clf.predict(lbp_val)) # mrelbp_feats['mrelbpVal']
    confusion = cm(np.asarray(y_val), predictions[-1])
    print(confusion)
    print(acc(np.asarray(y_val), predictions[-1]))

np.savetxt('mrelbp_multiscale1248Samples8xr_hsv_svm_c100_predictions_10-87.csv', predictions)
# np.savetxt('y_test_4-228.csv', y_val)
#
data_dic = {'x_train': x_train, 'x_val': x_val, 'y_train': y_train, 'y_val': y_val}
savemat('soybean_data_10-87.mat', data_dic)
