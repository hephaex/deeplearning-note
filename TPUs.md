Tensor Processing Units (TPUs) are Google’s custom-developed ASICs used to accelerate machine-learning workloads. You can run your training jobs on Cloud Machine Learning Engine, using Cloud TPU. Cloud ML Engine provides a job management interface so that you don't need to manage the TPU yourself.

https://github.com/tensorflow/tpu

Benchmarking Google’s new TPUv2
 - https://blog.riseml.com/benchmarking-googles-new-tpuv2-121c03b71384
https://github.com/UCSBarchlab/OpenTPU

## Samples:
* [Tensorflow: ResNet](training/resnet) - Using the ResNet-50 dataset with Cloud TPUs on ML Engine.
* [Tensorflow: HP Tuning - ResNet](hptuning/resent-hptuning) - How to run hyperparameter tuning jobs on Cloud Machine Learning Engine with Cloud TPUs using TensorFlow's tf.metrics.
* [Tensorflow: Hypertune - ResNet](hptuning/resent-hypertune) - How to run hyperparameter tuning jobs on Cloud Machine Learning Engine with Cloud TPUs using the cloudml-hypertune package.
* [Tensorflow: Templates](templates) - A collection of minimal templates that can be run on Cloud TPUs on Compute Engine, Cloud Machine Learning, and Colab.

If you’re looking for samples for how to use Cloud TPU, check out the guides here. 

Note: These guides do not use ML Engine
* [MNIST on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/mnist)
* [ResNet-50 on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/resnet)
* [Inception on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/inception)
* [Advanced Inception v3 on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/inception-v3-advanced)
* [RetinaNet on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/retinanet)
* [Transformer with Tensor2Tensor on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/transformer)
* [lSTM with TPU](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb)
