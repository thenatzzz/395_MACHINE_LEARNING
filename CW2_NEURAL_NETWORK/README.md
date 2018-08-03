Intructions:

The test function is located in src/test.py

Libraries needed to be imported
1. PIL and imread : for reading images
2. pickle : for turning object in python to pickle file
3. keras : to load the convolutional neural network

- To load a different .mat dataset files, change argument in loadmat
ex.
      path = "/your/image/file"

- To load the neural network, set pkl_net to the pickle file containing the
network. The default is set to point to the pickle file containing the network

      pkl_net = "net.pkl"

- To load the Convolutional neural network, set the path of the model to the
json file and weights to the .h5 file. The default paths are set to load the
model in the folder

    model = "model.json"
    weights = "model.h5"
