# TensorFlow 설치
* OS: ubuntu 14.04 LTS
* SW: python 2.7.x


## 1. Java8 설치

* webupd8team 이용하는 방법 

>$ sudo add-apt-repository ppa:webupd8team/java
>$ sudo apt-get update
>$ sudo apt-get install oracle-java8-installer

* 다운로드 받아서 설치하는 방법
 [오라클](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

## 2. Bazel 설치
* Bazel은 Google이 개발한 빌드 도구입니다. 

* Bazel 다운로드: [링크](https://github.com/bazelbuild/bazel/releases)
* Bazel 설치

>$ chmod +x bazel-version-installer-os.sh
>$ ./bazel-version-installer-os.sh --user


## 3. TensorFlow 설치

* TensorFlow 소스 다운로드 (git clone)

>$ /home/dwlee/softwares
>$ git clone -b master --recurse-submodules https://github.com/tensorflow/tensorflow.git

* Swig 설치

>$ sudo apt-get install swig

* Configure (ex. Not support GPU)

>$ ./configure
>Please specify the location of python. [Default is /home/dwlee/.python_virtual_envs/py27/bin/python]:
>Do you wish to build TensorFlow with GPU support? [y/N] N
>No GPU support will be enabled for TensorFlow
>Configuration finished

* Bazel로 빌드 

>$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
>$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
>$ pip install /tmp/tensorflow_pkg/<xxx.whl>

* TensorBoard 복사

"<TENSORFLOW_GIT_HOME>/bazel-bin/tensorflow/tensorboard" 에 있는 tensorboard 실행 스크립트 파일과 관련 디렉토리를 복사

e.g.) 
실행 스크립트 파일
$ cp ~/softwares/tensorflow/bazel-bin/tensorflow/tensorboard/tensorboard ~/.python_virtual_envs/py27/bin/

실행에 필요한 디렉토리
$ cp -rf ~/softwares/tensorflow/bazel-bin/tensorflow/tensorboard/tensorboard.runfiles ~/.python_virtual_envs/py27/bin


* TensorBoard 테스트

$ tensorboard --logdir=<log가 있는 디렉토리의 절대경로>

