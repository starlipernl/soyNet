# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.experimental import CosineDecayRestarts
import numpy as np
import pandas as pd
import cv2
import csv
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, DenseNet201
# from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.utils import Sequence
# install imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.keras import balanced_batch_generator


class BalancedCropDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling
       Class for keras data generator with balanced classes
    """
    def __init__(self, x, y, datagen, batch_size=32, crop_length=224, resize=300, balance=True):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape
        self.crop_length = crop_length
        self.resize = resize
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y,
                                                                  sampler=RandomOverSampler() if balance else None,
                                                                  batch_size=self.batch_size)

    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    def __len__(self):
        return self._shape[0] // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        x_crops = np.zeros((x_batch.shape[0], self.crop_length, self.crop_length, 3))
        for i in range(x_batch.shape[0]):
            resized_img = cv2.resize(x_batch[i], dsize=(self.resize, self.resize), interpolation=cv2.INTER_AREA)
            x_crops[i] = self.random_crop(resized_img, (self.crop_length, self.crop_length))
        # x_crops = x_crops.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_crops, y_batch, batch_size=self.batch_size).next()


def remove_bad_annotations(annotation_file):
    """Utility function to remove annotations that
       file does not exist"""
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data', 'allData')
    annotations = pd.read_csv(annotation_file)
    file_good = []
    for sample in annotations.itertuples():
        fname = os.path.join(data_dir, sample.fullID)
        if os.path.exists(fname):
            file_good.append(1)
        else:
            file_good.append(0)
    drop_idx = [idx for idx, val in enumerate(file_good) if val == 0]
    annotations.drop(drop_idx, inplace=True)
    annotations.reset_index(inplace=True, drop=True)
    annotations.to_csv('datay_val_goodFiles_11132020.csv')


def load_data(annotation_file, option, matchstring):
    """utility function to load data according to 3 different options
       and a matchstring
       option (str): plot - train/test split with specific
                      maize plot specified in matchstring as test data
               date - train/test split with specific date
                      as test data
               csv filename - use specified csv file as test annotations
       matchstring(str): matchstring to specify option keyword (10-87)
    """
    # Load annotation data
    annotations = pd.read_csv(annotation_file)
    if os.path.splitext(option)[1] == '.csv':
        test_annotations = pd.read_csv(option)
        option = 'traintest'

    # load data
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data', 'allData')
    data_train = []
    labels_train = []
    train_list = []
    data_test = []
    labels_test =[]
    test_list = []
    file_good = []

    if option.lower() == 'plot':
        for sample in annotations.itertuples():
            fname = os.path.join(data_dir, sample.fullID)
            print(fname)
            if sample.treatment_camera == matchstring:
                data_test.append(cv2.imread(fname))
                labels_test.append(sample.Annotation)
                train_list.append(sample.fullID)
            else:
                data_train.append(cv2.imread(fname))
                labels_train.append(sample.Annotation)
                test_list.append(sample.fullID)
    elif option.lower() == 'date':
        for sample in annotations.itertuples():
            fname = os.path.join(data_dir, sample.fullID)
            print(fname)
            if sample.orig_file_name[0:5] == matchstring:
                data_test.append(cv2.imread(fname))
                labels_test.append(sample.Annotation)
                train_list.append(sample.fullID)
            else:
                data_train.append(cv2.imread(fname))
                labels_train.append(sample.Annotation)
                test_list.append(sample.fullID)
    elif option.lower() == 'traintest':
        for sample in annotations.itertuples():
            fname = os.path.join(data_dir, sample.fullID)
            print(fname)
            data_train.append(cv2.imread(fname))
            labels_train.append(sample.Annotation)
        for sample in test_annotations.itertuples():
            fname = os.path.join(data_dir, sample.fullID)
            print(fname)
            data_test.append(cv2.imread(fname))
            labels_test.append(sample.Annotation)

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    print('Saving Files')
    with open('trainList.csv', 'w') as filehandle:
        csvwriter = csv.writer(filehandle)
        csvwriter.writerows(train_list)
    with open('testList.csv', 'w') as filehandle:
        csvwriter = csv.writer(filehandle)
        csvwriter.writerows(test_list)
    np.save(os.path.join(data_dir, 'x_train_' + option +'_'+matchstring + '.npy'), data_train)
    np.save(os.path.join(data_dir,'x_test_' + option +'_'+matchstring + '.npy'), data_test)
    np.save(os.path.join(data_dir, 'y_train_' + option +'_'+matchstring + '.npy'), labels_train)
    np.save(os.path.join(data_dir,'y_test_' + option +'_'+matchstring + '.npy'), labels_test)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('WARNING')

    # hyper-parameters
    epochs = 200
    batch_size = 32
    learning_rate = 1e-5
    dropout_rate1 = 0.5
    dropout_rate2 = 0.5
    lr_decay_steps = 50000
    downsize = 300  # downsize images before cropping
    width = 224  # for cropping
    height = 224  # for cropping
    input_shape = (width, height, 3)
    option = 'plot'
    matchstring = '10-87'

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data', 'allData')


    # Load arrays if needed
    print('Loading Data')
    x_train = np.load(os.path.join(data_dir, 'x_train_' + option +'_'+matchstring + '.npy'))
    x_val = np.load(os.path.join(data_dir,'x_test_' + option +'_'+matchstring + '.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_' + option +'_'+matchstring + '.npy'))
    y_val = np.load(os.path.join(data_dir,'y_test_' + option +'_'+matchstring + '.npy'))

    # We decided to merge classes 4 and 5
    y_train[y_train == 5] = 4
    y_val[y_val == 5] = 4
    print(x_train.shape, x_val.shape)
    #print(y_train)
    #print('Any NaN in Train: ', np.any(np.isnan(x_train)))
    #print('Any NaN in Val: ', np.any(np.isnan(x_val)))

    # images pre-processing and generate training data using datagenerators
    train_datagen = ImageDataGenerator(rescale=1./255.0,
                                       rotation_range=15,
                                       width_shift_range=0,  # shifts the image width wise
                                       height_shift_range=0,  # shifts the image height wise
                                       shear_range= 0.01,        # shears the image
                                       zoom_range=[0.8, 1.2],  # zoom the image
                                       horizontal_flip=True,    # flip the image horizontally
                                       vertical_flip=False,
                                       fill_mode='reflect',
                                       data_format='channels_last',
                                       brightness_range=[0.8, 1.2])  # [0.5, 1.3])  # changes the brightness level of the image

    # do not want to augment the validation data
    vali_datagen = ImageDataGenerator(rescale=1./255.0,
                                    rotation_range=0,
                                    width_shift_range=0,  # shifts the image width wise
                                    height_shift_range=0,  # shifts the image height wise
                                    shear_range=0.0,        # shears the image
                                    zoom_range=0.0, # [0.8, 1.25],  # zoom the image
                                    horizontal_flip=False,    # flip the image horizontally
                                    vertical_flip=False,
                                    fill_mode='reflect',
                                    data_format='channels_last',
                                    brightness_range=None)  # [0.5, 1.3])  # changes the brightness level of the image

    train_generator_balanced = BalancedCropDataGenerator(x_train, y_train, train_datagen, batch_size=batch_size,
                                                         crop_length=width, resize=downsize)
    train_steps = train_generator_balanced.steps_per_epoch
    validation_generator_balanced = BalancedCropDataGenerator(x_val, y_val, vali_datagen, batch_size=batch_size,
                                                              crop_length=width, resize=downsize, balance=False)
    valid_step = validation_generator_balanced.steps_per_epoch
    print('Validation Steps ', valid_step)

    # Preproces validation data if not using generator
    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    # random crop testing
    x_test = np.zeros((x_val.shape[0], width, height, 3))
    for idx, img in enumerate(x_val):
        testimg = np.array(img)
        testimg = cv2.resize(testimg, dsize=(downsize, downsize), interpolation=cv2.INTER_AREA)
        x_test[idx, ...] = random_crop(testimg, (width, height))/255.0

    # only if using unbalanced datagenerators
    # class_weights = class_weight.compute_class_weight(
    #                'balanced',
    #                 np.unique(y_train),
    #                 y_train)

    print('Creating Model')
    # load pre-trained base model
    base_model = DenseNet121(input_shape=input_shape, weights='imagenet', include_top=False)

    # DenseNet found to have better results than ResNet
    # base_model = ResNet152V2(input_shape=input_shape, weights='imagenet', include_top=False)
    # base_model.trainable = True
    # Found through experiments that the best results were with fine tuning the entire network
    # set only the layers after convolutional blocks 4 and 5 to be trainable
    # for layer in base_model.layers[0:141]:
    #     layer.trainable = False
    #    layer.trainable = True

    # build custom top layers
    base_output = base_model.output
    base_output = GlobalAveragePooling2D()(base_output)

    dense1 = Dense(512, activation='relu')(base_output)
    dropout1 = Dropout(dropout_rate1)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(dropout_rate2)(dense2)

    # output layer
    output_layer = Dense(5, activation='softmax')(dropout2)

    # full model
    model = Model(inputs=base_model.input, outputs=output_layer)

    # exponential decay learning rate scheduler found to give best results
    lr_sched = ExponentialDecay(
        learning_rate,
        lr_decay_steps,
        0.9,
        staircase=False)

    # # could not get good results with cosine decay, may need more tuning?
    # lr_cos = CosineDecayRestarts(
    #     learning_rate, lr_decay_steps, t_mul=1.0, m_mul=2.0, alpha=0.0,
    #     name=None
    # )

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=SGD(learning_rate=lr_sched, momentum=0.9, nesterov=True),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()

    # check which layers are trainable
    # for l in model.layers:
    #     print(l.name, l.trainable)

    # training model
    # specify model name for saving checkpoints
    modelName = option+'_'+ matchstring+'_'+'_dense121_balancedOverSample_SGD_GlobAvgPool_dense512_epoch'+str(epochs) \
                + '_batchSZ'+str(batch_size)+'_LR_' + str(learning_rate) + 'decay' + str(lr_decay_steps) \
                + '_Dropout'+str(int(dropout_rate1*100))+'_'+str(int(dropout_rate2*100)) + '_holdoutVal'
    print(modelName+'.h5\n')

    checkpoint = ModelCheckpoint('checkpoints/bestval_' + modelName + '.h5', monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    checkpoint_train = ModelCheckpoint('checkpoints/besttrain_' + modelName + '.h5', monitor='accuracy', verbose=1,
                                       save_best_only=True, mode='auto', period=1)
    history = model.fit(
            train_generator_balanced,
            epochs=epochs,
            validation_data=(x_test, y_val), # validation_generator_balanced,
            shuffle=True,
            callbacks=[checkpoint, checkpoint_train]
            )
    model.save('checkpoints/last_epoch_' + modelName + '.h5')


    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)  # convert training history to pandas dataframe

    # save to csv:
    hist_csv_file = os.path.join('history', modelName+'.csv')
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    # results using latest model (last epoch)
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    np.savetxt('results/results_lastepoch_' + modelName + '.csv', predictions, delimiter=',')
    print('Last Epoch Model Results: Accuracy = {:4.2f} | One-Off Accuracy = {:4.2f} | RMSE = {:4.2f} | MAE = {:4.2f}'.format(
          np.mean(np.equal(predictions, y_val)),
          np.mean(np.greater_equal(predictions,  y_val - 1) & np.less_equal(predictions, y_val + 1)),
          np.sqrt(np.mean((predictions-y_val)**2)),
          np.mean(np.abs(predictions-y_val))))

    # load best validation model
    modelTest = load_model("checkpoints/bestval_" + modelName + '.h5')
    predictions=modelTest.predict(x_test)
    predictions=np.argmax(predictions, axis=1)
    print(predictions)
    np.savetxt('results/results_bestval_' + modelName + '.csv', predictions, delimiter=',')
    print('Best Val Model Results: Accuracy = {:4.2f} | One-Off Accuracy = {:4.2f} | RMSE = {:4.2f} | MAE = {:4.2f}'.format(
          np.mean(np.equal(predictions, y_val)),
          np.mean(np.greater_equal(predictions,  y_val - 1) & np.less_equal(predictions, y_val + 1)),
          np.sqrt(np.mean((predictions-y_val)**2)),
          np.mean(np.abs(predictions-y_val))))

    # load best train checkpoint
    modelTest = load_model("checkpoints/besttrain_" + modelName + '.h5')
    predictions=modelTest.predict(x_test)
    predictions=np.argmax(predictions, axis=1)
    print(predictions)
    np.savetxt('results/results_besttrain_' + modelName + '.csv', predictions, delimiter=',')
    print('Best Train Model Results: Accuracy = {:4.2f} | One-Off Accuracy = {:4.2f} | RMSE = {:4.2f} | MAE = {:4.2f}'.format(
          np.mean(np.equal(predictions, y_val)),
          np.mean(np.greater_equal(predictions,  y_val - 1) & np.less_equal(predictions, y_val + 1)),
          np.sqrt(np.mean((predictions-y_val)**2)),
          np.mean(np.abs(predictions-y_val))))
