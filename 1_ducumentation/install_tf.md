# TensorFlow install on Ubuntu 20.04

1. wget http://developer.download.nvidia.com/…/cuda-repo-ubuntu2004…
2. sudo dpkg -i cuda-repo-ubuntu2004_8.0-18_amd64.deb
3. sudo apt-get update && sudo apt-get upgrade -y
4. sudo apt-get install -y opencl-headers build-essential protobuf-compiler libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev libopencv-core-dev libopencv-highgui-dev libsnappy-dev libsnappy1 libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0 libgoogle-glog-dev libgflags-dev liblmdb-dev git python-pip gfortran
5. sudo apt-get clean
6. sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`
7. sudo apt-get install -y cuda
8. tar xvzf cudnn-8.0-linux-x64-v5.0-rc.tgz
9. sudo cp cuda/include/cudnn.h /usr/local/cuda/include
10. sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
11. sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
12. nvidia-smi
13. sudo apt-get install python3-pip python3-dev
14. sudo pip install --upgrade https://storage.googleapis.com/…/tensorflow-1=2.6.0-cp38-n…
15. git clone https://github.com/hephaex/deeplearning-note
16. cd deeplearning-note/
