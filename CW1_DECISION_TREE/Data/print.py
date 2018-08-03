import numpy as np
import scipy.io
mat = scipy.io.loadmat('cleandata_students.mat')
print mat['y']
# change y to 1s and 0s
print np.log2(2)
