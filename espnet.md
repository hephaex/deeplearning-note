```
nvidia-docker run -i --rm --name  espnet_gpu0_20210315T0801 \
-v /home/mare/espnet/egs:/espnet/egs \
-v /home/mare/espnet/espnet:/espnet/espnet \
-v /home/mare/espnet/test:/espnet/test \
-v /home/mare/espnet/utils:/espnet/utils \
hephaex/espnet:gpupy37-cuda11.0
/bin/bash -c 
'cd /espnet/egs/an4/asr1; ./run.sh --verbose 1 --backend chainer --ngpu 1 --stage 3 --tag train_nodev_chainer_cuda11.0'
```
