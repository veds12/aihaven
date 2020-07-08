# -*- coding: utf-8 -*-
"""layers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fCQ_zLCcWNzgE99LK9B2cWrql8J3HgBO
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class gcn_layer(nn.Module):
  def __init__(self, ip_size, op_size):
    super(gcn_layer, self).__init__()
    self.ip_size = ip_size      # number of features for each node in the input
    self.op_size = op_size      # number of features for each node in the output
    self.weights = Parameter(torch.rand(self.ip_size, self.op_size, dtype = torch.float32, requires_grad = True))
    

  def compute(self, admat, features):

    ''' Forward Propagation through the layer according to the spectral rule '''
             
    self.D = torch.diag(admat.sum(1), diagonal = 0)
    self.out = torch.empty (admat.size[0], op_size)
    self.a_hat = admat + torch.eye(admat.size[0])    # Counting the contribution of each node to itself
    self.D_inv = self.D**(-0.5)
    self.a_hat = self.D_inv * self.a_hat * self.D_inv  # Normalising according to the spectral rule
    self.out = torch.dot(torch.dot(self.a_hat, features), self.weights)   # Forward propagate trhough the layer
    return self.out
