import numpy as np
import os

from src.fcnet import FullyConnectedNet
from keras.models import load_model
# import src.fcnet
from src.utils.solver import Solver
from src.utils.plot import plotGraphs
import src.utils.analysis as an
# import pickle as pkl

from PIL import Image
from numpy import genfromtxt
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
import keras
from keras.models import model_from_json
import src.utils.analysis as an

def dir_to_testset(glob_files):
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    return np.array(dataset)

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_batch_test(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        number_pic = X.shape[0]
        X = X.reshape(number_pic,1,48,48).transpose(0,2,3,1).astype("float")
        return X

def load_test(ROOT):
    Xte = load_batch_test(os.path.join(ROOT, 'test_batch.pkl'))
    return  Xte

def get_data_test(img_folder, num_test=1000,subtract_mean=True):
    X_test= load_test(img_folder)
    # Subsample the data
    mask = list(range(num_test))
    X_test = X_test[mask]
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_test, axis=0)
        X_test -= mean_image
    # Transpose so that channels come first
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    # Package data into a dictionary
    return {
      'X_test': X_test,
    }

def test_fer_model(img_folder, model="/path/to/model"):
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the file name of the images) and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ################## LOAD DATA ######################################
    number_pics = len([name for name in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, name))])
    img_folder_temp = img_folder + "/*.jpg"
    # Need to specify the image folder
    Data = dir_to_testset(img_folder_temp)
    #_____________________ OUTPUT to PKL FILE _____________________________
    dataset = dict()
    dataset['data'] = Data

    # Output to Pickle File
    output = open('test_batch.pkl','wb')

    pickle.dump(dataset,output,-1)
    output.close()
    PATH_TO_PKL = os.getcwd()
    data = get_data_test(PATH_TO_PKL,number_pics, subtract_mean=True)
    ################## READ PKL FILE ##################################
    pk = open(model, 'rb')

    test = pickle.load(pk)
    print(test)
    ################## TEST NET ########################################
    probs = test.loss(data['X_test'], None)
    print(probs)

    preds = an.getClassifications(probs)
    print(preds)
    return preds

def test_deep_fer_model(img_folder, json_model="/path/to/model", weights="/path/to/weights"):
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the file name of the images) and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ### Start your code here
    number_pics = len([name for name in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, name))])
    img_folder_temp = img_folder + "/*.jpg"
    # Need to specify the image folder
    # img_folder = "/neuralnets/src/test"
    Data = dir_to_testset(img_folder_temp)
    #_____________________ OUTPUT to PKL FILE _____________________________
    dataset = dict()
    dataset['data'] = Data
    # Output to Pickle File
    output = open('test_batch.pkl','wb')

    pickle.dump(dataset,output,-1)
    output.close()
    PATH_TO_PKL = os.getcwd()
    data = get_data_test(PATH_TO_PKL,number_pics, subtract_mean=True)
    X_test = data['X_test'].reshape(data['X_test'].shape[0], 48, 48, 1)
    ################## READ PKL FILE ##################################
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)
    print("Loaded model from disk")

    ################## TEST NET ########################################
    preds = loaded_model.predict(X_test,verbose = 0)
    matrix_shape = preds.shape
    no_of_rows = matrix_shape[0]

    return_array = an.getClassifications(preds)

    print(return_array)
    ### End of code
    return return_array

#TEST FCNET
path = "path/to/image"
pkl_net = "/homes/gw17/neuralnets/net.pkl"
test_fer_model(path,pkl_net)


#TEST CNN
model = "/homes/wt814/machinelearning/optional/neuralnets/model.json"
weights = "/homes/wt814/machinelearning/optional/neuralnets/model.h5"
test_deep_fer_model(path,model, weights)
