## The steps for centralized synchronous data parallelism are as follows:
1. A parameter server is used as the ground truth for the model weights. The weights are duplicated into multiple processes running on different hardware (GPUs on the same machine or on multiple machines).
2. Each duplicate model receives a different data mini-batch, and they independently go through the forward pass and backward pass where the gradients get computed.
3. The gradients are sent to the parameter server where they get averaged once they are all received. The weights get updated in a gradient descent fashion and the new weights get broadcast back to all the worker nodes.  

## This process is called "centralized" where the gradients get averaged. 
Another version of the algorithm can be "decentralized" where the resulting model weights get averaged: 
1. A master process broadcasts the weights of the model.
2. Each process can go through multiple iterations of the forward and backward passes with different data mini-batches. At this point, each process has very different weights.
3. The weights get sent to the master process, they get averaged across processes once they get all received, and the averaged weights get broadcast back to all the worker nodes.
The decentralized approach can be a bit faster because you don't need to communicate between machines as much, but it is not a proper implementation of the backpropagation algorithm. Those processes are synchronous because we need to wait for all the workers to finish their jobs. The same processes can happen asynchronously, only the gradients or weights are not averaged. You can learn more about it here: https://arxiv.org/pdf/2007.03970.pdf
When it comes to the centralized synchronous approach, Pytorch and TensorFlow seem to follow a slightly different strategy (https://pytorch.org/docs/stable/notes/ddp.html) as it doesn't seem to be using a parameter server as the gradients are synchronized and averaged on the worker processes. This is how the Pytorch DistributedDataParallel module is implemented (https://pytorch.org/.../torch.nn.parallel...), as well as the TensorFlow MultiWorkerMirroredStrategy one (https://www.tensorflow.org/.../MultiWorkerMirroredStrategy). It is impressive how simple they made training a model in a distributed fashion!

## The decentralized approach can be a bit faster because you don't need to communicate between machines as much, but it is not a proper implementation of the backpropagation algorithm. 
Those processes are synchronous because we need to wait for all the workers to finish their jobs. The same processes can happen asynchronously, only the gradients or weights are not averaged. You can learn more about it here: https://arxiv.org/pdf/2007.03970.pdf
When it comes to the centralized synchronous approach, Pytorch and TensorFlow seem to follow a slightly different strategy (https://pytorch.org/docs/stable/notes/ddp.html) as it doesn't seem to be using a parameter server as the gradients are synchronized and averaged on the worker processes. This is how the Pytorch DistributedDataParallel module is implemented (https://pytorch.org/.../torch.nn.parallel...), as well as the TensorFlow MultiWorkerMirroredStrategy one (https://www.tensorflow.org/.../MultiWorkerMirroredStrategy). It is impressive how simple they made training a model in a distributed fashion!