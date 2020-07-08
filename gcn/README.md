# Graph Convolutional Networks

Frameworck used : PyTorch

## layers.py

```
 from layers import gcn_layer
 mygcn = gcn_layer(ipsize, opsize)
 out = mygcn.compute(A, features)
```

* ipsize - Number of features per node in the input
* opsize - Number of features per node in the output
* A - The adjacency matrix (torch tensor) representing the graph (without self loops)
* features - Feature matrix (torch tensor) of the graph nodes

### Attributes

* weights - Trainable parameter of the layer of size (ipsize x opsize)
* D - The diagonal normalising matrix
* compute(A, features) - Propagate through the layer. Returns the ouput numpy array

## Current Tasks 

- [x] Write a script defining a gcn layer
- [x] Define the model
- [ ] Integrate train.py in model.py

## References 

* https://arxiv.org/abs/1609.02907
* https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0
