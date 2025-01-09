## stabilityai, Stable Point Aware 3D (SPAR3D) 발표 (25.1.8, 출처: stabilityai website)
Introducing Stable Point Aware 3D: Real-Time Editing and Complete Object Structure Generation (8 Jan)
[Key Takeaways]
-안정적인 포인트 인식 3D(SPAR3D)는 단일 이미지에서 1초 이내에 3D 오브젝트를 실시간으로 편집하고 완전한 구조를 생성합니다.
-동종 최초의 아키텍처를 갖춘 SPAR3D는 정밀한 포인트 클라우드 샘플링과 고급 메시 생성을 결합하여 3D 에셋 생성에 대한 전례 없는 제어 기능을 제공합니다. 기반 기술에 대해 자세히 알아보려면 여기에서 연구 논문 전문을 읽어보세요.
-이 모델은 허용되는 Stability AI 커뮤니티 라이선스에 따라 상업적 및 비상업적 용도로 모두 무료로 사용할 수 있습니다.
-허깅 페이스에서 가중치를 다운로드하고 GitHub에서 코드를 다운로드하거나 Stability AI 개발자 플랫폼 API를 통해 모델에 액세스할 수 있습니다.
[Where the model excels] 
SPAR3D는 다음과 같은 고급 기능으로 게임 개발자, 제품 디자이너, 환경 제작자를 위한 3D 프로토타이핑을 혁신합니다:
-전례 없는 제어 기능: 사용자가 포인트 클라우드를 삭제, 복제, 늘이기, 기능 추가 또는 포인트 색상 변경을 통해 직접 편집할 수 있습니다.
-완벽한 구조 예측: 물체의 뒷면과 같이 일반적으로 숨겨진 영역을 포함하여 전체 360도 뷰에 대한 정확한 형상과 상세한 예측을 제공하여 3D 구조를 향상시킵니다.
-초고속 생성: 편집된 포인트 클라우드를 단 0.3초 만에 최종 메시로 변환하여 원활한 실시간 편집이 가능합니다. 단일 입력 이미지에서 오브젝트당 0.7초 만에 매우 상세한 3D 메시를 생성합니다.


# Various tools for model transformation
Th is able to convert models between frameworks and models on the framework. 
- **Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis**
   - https://twitter.com/_akhaliq/status/1741666796770390207
   - https://huggingface.co/papers/2312.17681
- **onnx-tensorflow** Tool to convert ONNX to TensorFlow
    - GitHub **https://github.com/onnx/onnx-tensorflow**
    - PyPI **https://pypi.org/project/onnx-tf/**
- **tensorflow-onnx** Tool to convert Tensorflow to ONNX
    - GitHub **https://github.com/onnx/tensorflow-onnx**
    - PyPI **https://pypi.org/project/tf2onnx/**
- **onnx2keras** A tool to convert ONNX to Keras
    - GitHub **https://github.com/nerox8664/onnx2keras**
    - PyPI **https://pypi.org/project/onnx2keras/**
- **tflite2onnx** Tool to convert TensorFlowLite to ONNX
    - HomePage **https://jackwish.net/tflite2onnx/**
    - GitHub **https://github.com/jackwish/tflite2onnx**
    - PyPI **https://pypi.org/project/tflite2onnx/**
- **sklearn-onnx** A tool to convert sklearn to ONNX
    - GitHub **https://github.com/onnx/sklearn-onnx**
- **onnx-mlir** Tool to convert ONNX to MLIR
    - GitHub **https://github.com/onnx/onnx-mlir**
- **keras-onnx** A tool to convert Keras to ONNX
    - GitHub **https://github.com/onnx/keras-onnx**
- **onnx-tensorrt** Tool to convert ONNX to TensorRT
    - GitHub **https://github.com/onnx/onnx-tensorrt**
- **onnx-coreml** Tool to convert ONNX to CoreML
    - GitHub **https://github.com/onnx/onnx-coreml**
- **onnx-simplifier** Tool for optimizing the ONNX model structure
    - GitHub **https://github.com/daquexian/onnx-simplifier**
    - PyPI **https://pypi.org/project/onnx-simplifier/**
- **OpenVINO Deep Learning Deployment Toolkit (DLDT) - Model Optimizer** Conversion of TensorFlow, ONNX, MXNet, and Caffe to OpenVINO IR format and other useful toolkits
    - HomePage **https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html**
    - GitHub **https://github.com/openvinotoolkit/openvino**
- **coremltools** Tools for converting various frameworks to CoreML models
    - GitHub **https://github.com/apple/coremltools**
    - PyPI **https://pypi.org/project/coremltools/**
- **MMdnn** a comprehensive and cross-framework tool to convert
    - GitHub **https://github.com/microsoft/MMdnn**
    - PyPI **https://pypi.org/project/mmdnn/**
- **torch2tflite** It uses ONNX and TF2 as bridge between Torch and TFLite'

    - GitHub **https://github.com/omerferhatt/torch2tflite**
- **openvino2tensorflow** Tool to convert OpenVINO IR models to TensorFlow
    - GitHub **https://github.com/PINTO0309/openvino2tensorflow**
    - PyPI **https://pypi.org/project/openvino2tensorflow/**
- **PINTO_model_zoo** Model collections for PyTorch (ONNX), Caffe, TensorFlow, TensorflowLite, CoreML, TF-TRT and TFJS. A large number of model conversion scripts have been committed
    - GitHub **https://github.com/PINTO0309/PINTO_model_zoo**
