import numpy as np
import json


with open('network/test_1_loss_batch.json') as data_file:
    data = json.load(data_file)

a = np.array(data[1:])

a.reshape(len(data[1:],3))

import matplotlib.pyplot as plt
plt.semilogy(a[:,1])
plt.semilogy(a[:,2])
plt.show()