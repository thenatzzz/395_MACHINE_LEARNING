from PIL import Image
from numpy import genfromtxt
# import gzip, cPickle
import gzip
import _pickle as cPickle
import pickle
from glob import glob
import numpy as np
import pandas as pd

from builtins import range
from six.moves import cPickle as pickle
import os
from scipy.misc import imread
import platform

def dir_to_dataset(glob_files, loc_train_labels=""):
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)

    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        print()
        return np.array(dataset), np.array(df["emotion"])
    else:
        return np.array(dataset)


number_train = 28709
number_test = 3589
NUMBER_PIC = number_train + number_test

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_FER2013_batch(filename):
    """ load single batch of FER2013 """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']

        if(X.shape[0] == number_train):
            number_pic = X.shape[0]
        elif (X.shape[0] == number_test):
            number_pic = X.shape[0]
        X = X.reshape(number_pic,1,48,48).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y
def load_FER2013(ROOT):
    """ load all of FER2013 """

    Xtr, Ytr = load_FER2013_batch(os.path.join(ROOT, 'train_batch'))

    Xte, Yte = load_FER2013_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_FER2013_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the FER2013 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw FER2013 data

    FER2013_dir = '/homes/nj2217/ML/neuralnets/src/fer2013-batches'


    X_train, y_train, X_test, y_test = load_FER2013(FER2013_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
