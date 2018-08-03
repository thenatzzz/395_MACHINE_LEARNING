import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = get_CIFAR10_data(49, 1, 0)

INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray([400,400])
NUM_CLASS = 10
net = FullyConnectedNet(HIDDEN_DIMS,INPUT_DIMS,NUM_CLASS)
solver = Solver(net, data,update_rule='sgd',optim_config={\
                'learning_rate': 1e-3},\
            num_epochs=20,\
            batch_size = 10,\
            print_every=1)
solver.train()


##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
