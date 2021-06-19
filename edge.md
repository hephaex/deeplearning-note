
# 마스크캠(착용감지)
MaskCam은 모든 AI 계산이 에지에서 수행되는 실시간으로 군중 얼굴 마스크 사용량을 측정하는 Jetson Nano 기반 스마트 카메라 시스템의 프로토 타입 참조 설계입니다. MaskCam은 시야에 있는 사람을 감지 및 추적하고 물체 감지, 추적 및 알고리즘을 통해 마스크를 착용하고 있는지 확인합니다. 웹 GUI를 사용하여 시야에서 안면 마스크 준수를 모니터링 할 수있는 클라우드에 통계 (비디오 아님)를 업로드합니다. 흥미로운 비디오 조각을 로컬 디스크에 저장하고 (예 : 마스크를 착용하지 않은 많은 사람들이 갑자기 유입 됨) RTSP를 통해 선택적으로 비디오를 스트리밍 할 수 있습니다.
  
* 공헌자: BDTI, TRYOLABS, JABIL
* 젯팩버젼: JetPack 4.4.1 or 4.5
* GitHub(소스코드): https://github.com/bdtinc/maskcam 
* 논문: https://www.bdti.com/.../Developing-Prototype-Mask-Jetson...

# Robottle
RPLidar로 SLAM을 사용하여 환경지도를 구성하고 Jetson Nano Board의 GPU에서 실행되는 Deep Neural Network를 사용하여 병을 감지하여 장애물이있는 임의의 환경에서 병을 수집 할 수있는 자율 로봇입니다. Robottle은 EPFL의 학술 대회를 위해 설계되었습니다. 로봇은 10 분 동안 병으로 가득 찬 경기장에서 자동으로 병을 수집하고 경기장의 한 구석 인 재활용 경기장으로 가져와야합니다.
* GitHub(소스코드): https://github.com/arthurBricq/ros_robottle
* 데모 동영상: https://youtu.be/XJpJSuhSZN4
* 논문: https://github.com/arth.../ros_robottle/blob/main/report.pdf
