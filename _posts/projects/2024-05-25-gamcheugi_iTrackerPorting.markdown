---
layout: post
title:  "iTracker 소스코드 분석 및 포팅"
date:   2024-05-25 01:15:16 +0900
categories: projects
tags: gamcheugi computervision
---

Appearance-Based GazeTracking Task에서 CNN 기반 접근법의 고전인 “EyeTracking for Everyone” 에서 제시되었던 모델인 iTracker를 직접 사용해보기로 하였다. 

집필진이 공개한 <a href=https://github.com/CSAILVision/GazeCapture>Github Repository</a>에서 관련 모델의 관련 스크립트 파일과 모델의 소스코드 (Caffe/Pytorch)를 모두 제공한다. 하지만 GazeCapture Dataset에서 Pytorch 데이터셋을 구성하고, Metadata를 생성하는 것 이외에는 신규 데이터 입력 등 관련 API가 제공되지 않아서 해당 모델을 AI 서비스에 사용하려면 custom API를 직접 만들어야 한다. 

<h3>간략한 모델 개요</h3>
iTracker는 다음과 같은 입력값을 처리한다. 

![alt text]({{"/assets/images/2024-05-25-iTrackerPorting/model_input.PNG" | relative_url}})  

AppleFace와 AppleLeftEye, AppleRightEye 모두 (X, Y, W, H) 형식으로 정의되는 원본 frame내의 bounding box로 정의되고, 모델에 입력 전 (224, 224) 크기로 resizing된다. 이때 Facial Features는 Apple's Face Detector Pipeline으로 측정하였다고 논문에 적었으나, 정확히 어떤 모듈이나 프로그램을 사용했는지는 알 수가 없었다.
정확도는 핸드폰의 경우 1.53cm의 Mean Error, 패드의 경우 2.38cm를 보였다. 패드의 경우에는 데이터셋에 상대적으로 패드에서 촬영한 프레임 개수가 현저히 적어서 그런 것으로 추정한다 (직접 확인해본 결과 약 7:1 정도의 비율이다). FaceGrid는 (25,25) 모양의 array로, 원본 프레임 내에서 얼굴의 상대적 위치를 표현한다. 

또한, y-label은 (x,y)로 2차원 좌표지만, 해당 좌표는 모바일 장비의 카메라를 원점으로 하는 가상의 plane 내의 좌표이기 때문에, 기종과 상관없이 동일한 기준으로 모델 평가가 가능하다는 장점이 있으나 실사용을 위해서는 해당 기기의 실제 비율과, 인치당 픽셀 수 등을 고려하여 픽셀 좌표로 매핑하는 작업이 필요하다. 

자체적인 입력 데이터 생성 스크립트는 모두 MATLAB으로 되어 있어, 사용하기 위해서는 파이썬으로 포팅하고, 결과가 원본과 같은지 확인해야 했다. 

<h3>포팅 작업<h3>
포팅 대상이 된 스크립트는 다음과 같다. 

1. cam2screen.m (Prediction Space 상의 (x,y) 좌표를 현재 기기의 화면 픽셀로 변환)  

2. faceGridFromFaceParams.m (Face Bounding Box의 params(X, Y, W, H)를 참고하여 25,25짜리 FaceGrid array를 생성함)  

Single sample inference를 위해 필요한 추가 컴포넌트는 다음과 같다:

3. infer.py (frame, y-label(X,Y)을 입력했을 때 Prediction Space Coordinates 반환)

그리고 이외에 정상 작동을 위해 많은 함수 등을 원본 소스코드를 참조하여 리팩토링하거나 역설계하여 파이썬 모듈화하였다. 


<h3>포팅 작업 시 발견한 문제점 및 해결 전략<h3>
<h4> 1. FaceGridFromFaceParams.m 포팅 중 원본 데이터 재현 실패 문제 <h4>
[문제]
FaceGridFromFaceParams.m 스크립트를 파이썬으로 포팅했을 때 원본 데이터의 FaceGrid를 재현하지 못하는 버그를 관찰하였다. 소스코드를 관찰했을 때는 뭔가 Orientation과 관련 있는 문제인 것 같고, 좌표 기준점의 중간 변경 혹은 이미지를 스크립트에 입력하기 전 어떤 모종의 Preprocessing을 했을 수도 있다. 또한 원본 FaceGrid를 직접 원본 이미지 프레임에 overlay 했을 때 FaceGrid params에 나온 대로의 결과가 육안으로 관찰한 결과가 미묘하게 어긋나는 문제도 있었으나, 논문을 읽어봐도 Face Grid 생성 방법에 대해서는 설명이 거의 없었다. 
[해결]
Error Analysis를 통해 임시로 조치하였다. Orientation==1일 때 원본 데이터와 포팅한 스크립트의 결과값 간의 차이가 일정한 패턴에 따라 나타나는 것을 관찰하였다. 
따라서 Orientation==1인 샘플만 사용하고, 기계적 보정을 통해서 결국 원본 데이터와 똑같은 FaceGrid를 생성하도록 하였다. 이 문제는 추후 다시 재방문할 예정이다.   
<br>
<br>
<br>
<h4> 2. Apple Face Detector의 결과값 재현 문제 <h4>
[문제]
원본은 Apple Face Detector를 사용했다고 했지만 집필진이 데이터셋 구축 당시 사용했던 라이브러리를 정확히 알 수가 없어 컴포넌트를 대체해야 했다. 가장 유력한 것은 지금까지 사용해왔던 dlib이나 dlib의 front_face_detector가 원본과 동일한 Boundingbox를 생성할지는 미지수였기 때문에, 동일한 결과를 낼 수 있음을 보장하는 방안을 마련해야 했다. 

[해결(진행 중)]
먼저 Bounding Box가 원본과 구체적으로 "얼마나 유사한가"를 정량적으로 판단할 지표가 있어야 했다. 이 부분은 Intersection over Union을 사용하면 적당할 것 같았다. 
따라서 Intersection over Union을 계산하는 함수를 제작하고, 현재 전체 데이터셋에 다른 calibration 없이 이전에 사용하였던 dlib face detector와 eye cropping mechanism을 그대로 사용하여 patch를 생성했을 때 원본과 얼마나 유사한지 테스트해보았다. 

```python
mean Face IoU:  0.7304785684022552
mean leftEye IoU:  0.29717062573401426
mean rightEye IoU:  0.28481355962929844
```

즉 Face Patch에 비해 Eye Patch는 훨씬 원본과 다르다. 이는 육안으로 확인해봐도 원본 Eye Patch가 훨씬 더 넓은 영역을 포괄함을 알 수 있어 당연한 결과였다. 

![alt text]({{"/assets/images/2024-05-25-iTrackerPorting/bb_comparison.PNG" | relative_url}})  

대략 다음과 같은 느낌이다. <font color='green'>초록색 Boundingbox</font>가 <font color='green'>Dlib</font>,  <font color='red'>붉은색</font>이 데이터셋의 json 메타데이터를 열람해서 참조한  <font color='red'>Apple Face Detector Bounding box</font>들이다.

이제 dlib의 예측치를 보정하여 Apple Face Detector의 Cropping 결과물과 동일한 결과를 낼 수 있도록 하는 보정 알고리즘을 만들어야 한다. 

현재 생각한 방법은: 
    1. 가로 세로 padding을 임의로 늘이거나 줄이며 평균 IOU가 최대가 되는 지점을 찾는 padding을 찾아 이후 기계적 보정을 더한다.
    2. $(aX1, bY1, cW1, dH1) -> (X2, Y2, W2, H2)$ 로 1:1 매핑이 가능하다. Mean IOU를 Loss값으로 쓰는 Custom Loss Function을 만들어서 Polynomial Regression 모델로 
        최적 파라미터 조합 a, b, c, d를 찾는다. 

    3. 2번이 결과가 안 좋을 경우, FaceGrid까지 넣어본다. (이 경우 multiperceptron 구조 말고 다른 머신러닝 알고리즘이 있는가?)

