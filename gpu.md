```
$nvidia-smi
```

```
import tourch
print(torch.cuda.is_available())

torch.cuda.get_device_name(0)
```

```
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

```
# cat /etc/os-release
NAME="Ubuntu"
VERSION="16.04.2 LTS (Xenial Xerus)"
# lspci | grep -i nvidia
07:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
08:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
# cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  375.74  Wed Jun 14 01:39:39 PDT 2017
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4) 
```

```
# cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  375.74  Wed Jun 14 01:39:39 PDT 2017
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4)
```

```
# ls /usr/local/cuda/lib64/libcudnn*
/usr/local/cuda/lib64/libcudnn.so    /usr/local/cuda/lib64/libcudnn.so.5.1.10
/usr/local/cuda/lib64/libcudnn.so.5  /usr/local/cuda/lib64/libcudnn_static.a

# cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR
#define CUDNN_MAJOR      5
#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
# cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MINOR
#define CUDNN_MINOR      1
#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
# cat /usr/local/cuda/include/cudnn.h | grep CUDNN_PATCH
#define CUDNN_PATCHLEVEL 10
#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
```

```
# ls libcudnn.**
libcudnn.so  libcudnn.so.5  libcudnn.so.5.1.10  libcudnn_static.a
# mkdir ~/backup
# mv libcudnn* ~/backup/
# cd /usr/local/cuda-8.0/include/
# ls cudnn*
cudnn.h
# mv cudnn* ~/backup/
# ldconfig
```

```
# dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
Selecting previously unselected package libcudnn6.
(Reading database ... 215725 files and directories currently installed.)
Preparing to unpack libcudnn6_6.0.21-1+cuda8.0_amd64.deb ...
Unpacking libcudnn6 (6.0.21-1+cuda8.0) ...
Setting up libcudnn6 (6.0.21-1+cuda8.0) ...
Processing triggers for libc-bin (2.23-0ubuntu9) ...
# dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
Selecting previously unselected package libcudnn6-dev.
(Reading database ... 215731 files and directories currently installed.)
Preparing to unpack libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb ...
Unpacking libcudnn6-dev (6.0.21-1+cuda8.0) ...
Setting up libcudnn6-dev (6.0.21-1+cuda8.0) ...
update-alternatives: error: unable to read link '/etc/alternatives/libcudnn': Invalid argument
update-alternatives: error: no alternatives for libcudnn
dpkg: error processing package libcudnn6-dev (--install):
 subprocess installed post-installation script returned error exit status 2
Errors were encountered while processing:
 libcudnn6-dev
 
# ls -l  /etc/alternatives/libcudnn
-rw-r--r-- 1 root root 0 Apr  5 11:22 /etc/alternatives/libcudnn
# cat /etc/alternatives/libcudnn
```

```
# rm /etc/alternatives/libcudnn
# ln -s /usr/include/x86_64-linux-gnu/cudnn_v6.h /etc/alternatives/libcudnn
# ls -l /etc/alternatives/libcudnn
lrwxrwxrwx 1 root root 40 Sep 21 07:16 /etc/alternatives/libcudnn -> /usr/include/x86_64-linux-gnu/cudnn_v6.h
# dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
(Reading database ... 215737 files and directories currently installed.)
Preparing to unpack libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb ...
Unpacking libcudnn6-dev (6.0.21-1+cuda8.0) over (6.0.21-1+cuda8.0) ...
Setting up libcudnn6-dev (6.0.21-1+cuda8.0) ...
update-alternatives: warning: /etc/alternatives/libcudnn has been changed (manually or by a script); switching to manual updates only
update-alternatives: warning: forcing reinstallation of alternative /usr/include/x86_64-linux-gnu/cudnn_v6.h because link group libcudnn is broken
# dpkg -l | grep cudnn
ii  libcudnn5                                         5.1.10-1+cuda8.0                           amd64        cuDNN runtime libraries
ii  libcudnn6                                         6.0.21-1+cuda8.0                           amd64        cuDNN runtime libraries
ii  libcudnn6-dev                                     6.0.21-1+cuda8.0                           amd64        cuDNN development libraries and headers
```

```
# cat /usr/local/cuda/version.txt
CUDA Version 8.0.61
# /usr/local/cuda-8.0/bin/nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```
