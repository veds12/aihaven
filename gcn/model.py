# Author : Vedant Shah
# E-mail : vedantshah2012@gmail.com

import torch.nn as nn
import torch.nn.functional as F

from layers import gcn_layer

# A two layer Graph Convolutional network


class gcn(nn.Module):
    def __init__(self, ipi, opi, opo):
        super(gcn, self).__init__()
        self.layer1 = gcn_layer(ipi, opi)
        self.layer2 = gcn_layer(opi, opo)

    def forward(self, admat, features):
        features = F.ReLu(self.layer1.compute(admat, features))
        features = F.softmax(self.layer2.compute(admat, features), dim=1)
        return features
