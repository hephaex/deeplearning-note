# Black-Box Adversarial Perturbations
Implementation of [Simple Black-Box Adversarial Perturbations for Deep Networks,https://arxiv.org/abs/1612.06299](https://arxiv.org/abs/1612.06299) in Keras
fork from [link](https://github.com/iamgroot42/Simple-Black-Box-Adversarial-Perturbations-for-Deep-Networks)
* `python cifar100.py` to train a basic CNN for cifar100 and save that file.
* `python find_better.py <model>` to go through cifar100 test dataset and find a good image (as defined in the paper).
* `python per.py <KERAS_MODEL> <IMAGE_in_NUMPY>` : currently works for cifar images only. 
