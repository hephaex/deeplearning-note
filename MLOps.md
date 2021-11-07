# 
MLOps 란 무엇입니까?

공헌자(저자): RICK MERRITT
원문: https://blogs.nvidia.com/blog/2020/09/03/what-is-mlops/
저자 내용중에서 순서에 상관없이 중간 부분을 요약 해봅니다.
...
NVIDIA의 AI 인프라 디렉터인 Nicolas Koumchatzky는 "가능한 한 오픈 소스 코드를 사용하려고 했지만 대규모로 수행하려는 작업에 대한 솔루션이 없는 경우가 많았습니다."
...
...
MLOps는 엔터프라이즈 애플리케이션을 효율적으로 작성, 배포 및 실행하는 최신 방식인 DevOps의 기존 분야를 모델로 합니다. DevOps는 10년 전에 소프트웨어 개발자(Devs)와 IT 운영 팀(Ops)의 운영중인 부족을 협력할 수 있는 방법으로 시작되었습니다.
MLOps는 데이터 세트를 선별하고 이를 분석하는 AI 모델을 구축하는 데이터 과학자를 팀에 추가합니다. 여기에는 규칙적이고 자동화된 방식으로 모델을 통해 해당 데이터 세트를 실행하는 ML 엔지니어도 포함됩니다.
이는 엄격한 관리뿐만 아니라 성과에서도 큰 도전입니다. 데이터 세트는 방대하고 증가하며 실시간으로 변경될 수 있습니다. AI 모델은 실험, 조정 및 리트레이닝 주기를 통해 신중하게 추적해야 합니다.
...
따라서 MLOps는 기업의 성장에 따라 확장할 수 있는 강력한 AI 인프라가 필요합니다. 이를 위해 많은 회사에서 NVIDIA DGX 시스템, CUDA-X 및 NVIDIA 소프트웨어 허브인 NGC에서 사용할 수 있는 기타 소프트웨어 구성 요소를 사용합니다.
NVIDIA의 Koumchatzky 팀은 자율주행 차량을 만들고 테스트하기 위한 플랫폼인 NVIDIA DRIVE 를 호스팅하는 MLOps 소프트웨어인 MagLev를 개발했습니다. MLOps 기반의 일부로 NVIDIA에서 개발한 구성 요소 집합인 NVIDIA Container Runtime 및 Apollo를 사용하여 거대한 클러스터에서 실행되는 Kubernetes 컨테이너를 관리하고 모니터링합니다.
NVIDIA는 파트너의 소프트웨어 외에도 DGX 시스템을 기반으로 하는 AI 인프라 관리를 위한 주로 오픈 소스 도구 모음을 제공하며 이것이 MLOps의 기반입니다. 이러한 소프트웨어 도구에는 다음이 포함됩니다.
개별 시스템 프로비저닝을 위한 Foreman 및 MAAS(Metal as a Service)클러스터 구성 관리를 위한 Ansible 및 Git 모니터링 및 보고를 위한 DCGM (Data Center GPU Manager) 및 NVSM(NVIDIA 시스템 관리) NVIDIA Container Runtime 은 GPU 인식 컨테이너를 시작하고 NVIDIA GPU Operator 는 Kubernetes에서 GPU 관리를 단순화합니다.
Triton Inference Server 및 TensorRT를 사용하여 프로덕션 환경에 AI 모델 배포 그리고 DeepOps 배포하는 방법에 대한 스크립트와 지침은 위의 모든 요소를 조율합니다.
