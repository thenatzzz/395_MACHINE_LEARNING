import cv2
import glob
import numpy as np
import pandas as pd
import os


def load_FER2013(ROOT):
    """ load all of FER2013 """
    X = np.load(os.path.join(ROOT, 'fer2013_data.npy'))
    Y = np.load(os.path.join(ROOT, 'fer2013_labels.npy'))
    X = X.transpose(0,1,2,3)
    Xtr = X[:28709]
    Ytr = Y[:28709]
    Xte = X[28709:]
    Yte = Y[28709:]
    return Xtr, Ytr, Xte, Yte


def get_FER2013_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):

    FER2013_dir = '/homes/wt814/machinelearning/optional/neuralnets'
    X_train, y_train, X_test, y_test = load_FER2013(FER2013_dir)

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]


    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
