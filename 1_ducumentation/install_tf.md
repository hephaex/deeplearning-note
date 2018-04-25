# TensorFlow install on Ubuntu 16.04

1. wget http://developer.download.nvidia.com/…/cuda-repo-ubuntu1404…
2. sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
3. sudo apt-get update
4. sudo apt-get upgrade -y
5. sudo apt-get install -y opencl-headers build-essential protobuf-compiler libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev libopencv-core-dev libopencv-highgui-dev libsnappy-dev libsnappy1 libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0 libgoogle-glog-dev libgflags-dev liblmdb-dev git python-pip gfortran
6. sudo apt-get clean
7. sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`
8. sudo apt-get install -y cuda
9. tar xvzf cudnn-7.5-linux-x64-v5.0-rc.tgz
10. sudo cp cuda/include/cudnn.h /usr/local/cuda/include
11. sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
12. sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
13. nvidia-smi
14. sudo apt-get install python3-pip python3-dev
15. sudo pip install --upgrade https://storage.googleapis.com/…/tensorflow-1.7.0-cp36-n…
16. git clone https://github.com/nlintz/TensorFlow-Tutorials
17. cd TensorFlow-Tutorials/
18. vi ~/.profile # add PATH, LD PATH
19. source ~/.profile
20. python 7_lstm.py
