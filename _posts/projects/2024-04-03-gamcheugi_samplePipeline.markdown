---
layout: post
title:  "얼굴 이미지 추출 스크립트 제작 과정"
date:   2024-04-03 01:15:16 +0900
categories: projects
tags: gamcheugi computervision
---
<link rel="stylesheet" href="{{ '/assets/styles/styles.css/' | relative_url }}">

<b>개요</b>  
모델 테스트 목적으로 자체적인 샘플 이미지를 생성하는 파이썬 스크립트를 제작하였다. 
이후에 프로젝트 서비스에서도 입력 단을 개발하는데 참고할 만한 프로토타입 역할도 할 수 있으리라 판단했다. 

구조는 웹캠에서 스트리밍되는 프레임 1개 당 현재 모니터의 Point of Gaze(PoG;현재 바라보고 있는 픽셀 좌표(X, Y))를
매핑하는 식으로 제작했다.   

특정 프레임에서 얼굴 영역을 찾기 위해서는 dlib 라이브러리에 내장되어 있는 fontral_face_detector를 사용했다. 
사용이 간편하고, 참고한 여러 논문에서 dlib의 face detector와 landmark estimator를 사전 데이터 처리에 사용하는 것을 봤기 때문이다.  
(이상하게도 pip을 통한 자동 설치가 안되에서 Precompile된 wheel을 따로 구해서 다운 받는 식으로 설치할 수 있었다). 

One-shot구조의 스크립트는 대략 다음과 같이 구조화했다.

![alt text]({{"/assets/images/2024-04-04-samplePipeline/1.png" | relative_url}})

각 단계에서 얻어지는 중간 산출물들을 보면 대략 다음과 같은 단계를 거친다. 
<h3>1</h3>
queue에서 특정 화면 위치의 점 좌표를 꺼내와 표시한다. 사용자가 Enter키를 누르면 그 시점의 프레임이 캡쳐된다.   
<br>
<br>
<br>
<h3>2</h3>  
<br>
![alt text]({{"/assets/images/2.png" | relative_url}})  
dlib의 frontal_face_detector()를 써서 현재 프레임에서 얼굴 영역을 찾는다.FaceEstimator 클래스를 통해 Dlib Functions들을 wrapping하도록 했고, Face Object를 통해서 Bounding box와 Landmark들을 객체 단위로 관리하도록 설계했다. 

<br>
<br>
<br>
<h3>3</h3>  
![alt text]({{"/assets/images/3.PNG" | relative_url}})  
dlib의 face_estimator()로 facial landmarks를 찾는다. 기본적으로 68개가 나오는데 특정 논문은 이 중에서 일부만 사용하는 경우가 있다. 예컨대 자체 CNN 모델의 참조 논문으로 사용한 “Convolutional Neural Network Based Technique for Gaze Estimation on Mobile Devices” 에서는 이중 눈과 눈썹, 턱 윤곽 영역에만 있는 약 39개 정도의 landmarks들만 선별해서 모델 입력값으로 사용된다.  
<br>
<br>
<br>
<h3>4</h3>  
![alt text]({{"/assets/images/4.PNG" | relative_url}})  
CNN 모델에 입력값으로 넣기 위해 사이즈를 통일해줘야 한다. 상술한 논문에서는 (244, 244, 3)으로 맞췄기 때문에 Bounding Box 기준으로 Cropping 후 Opencv의 Resizing으로 이미지를 처리해준다. 

이때 Facial landmarks의 좌표들도 image Transformation에 맞춰서 계속해서 align해줬다. Image transformation을 먼저 하고 landmark를 찾는 것이 훨씬 less intensive했겠지만, cropping이나 resizing을 하고 face_estimator를 사용하니까 검출률이 크게 떨어지는 문제가 있었다.  

그래서 좌표를 찾아놓고 -> 좌표를 transformation에 맞춰 재조정하는, 살짝 번거로운 처리 과정을 거치게 되었다.   

<h3>5</h3>  
![alt text]({{"/assets/images/5.PNG" | relative_url}})  

CNN 모델에 넣을 Landmark Image도 바로 간단하게 생성할 수 있다. Facial landmark에 좌표가 있으므로 244,244크기의
np.zeros(shape=(244, 244))로 initialize한 후 landmark 좌표에 해당하는 위치에만 +1 해주면 된다. 
논문에서는 얼굴의 각도를 모델이 고려할 수 있도록 하는 요소로 추정된다. 3D 모델링 구축을 먼저 하는 일부 논문에서는 얼굴의 yaw, pitch 등 3차원 공간 내에서의 Parameter를 입력값으로 사용한 경우도 있는데, 해당 논문은 이 과정을 간소화했다. 

그 과정에서 최대한 나름 모듈화를 적용해 체계적으로 샘플을 추출하는 시퀀스를 만들려고 했고, 그 과정에서 구조가 맘에 안 들어 리팩토링을 두 번 수행하느라 생각보다 시간이 더 걸렸다.

<h3>6</h3>
![alt text]({{"/assets/images/6.PNG" | relative_url}})  

dlib은 눈을 감은 경우에도 안정적으로 landmark 위치를 추정한다. 다만 눈을 감은 사진이 시선 추적 모델의 훈련 샘플로 사용될 순 없으므로, 샘플 추출 시퀀스에서는 눈을 감은 경우 재촬영을 하는 것이 바람직하다. dlib이나 opencv에서 자체적으로 눈을 감았는지 여부를 알려주지는 않으므로, 자체적인 알고리즘으로 눈을 감았는지 여부를 분별해야 한다. 

눈의 Eye Aspect Ratio를 통해 눈을 감았는지 여부를 측정하는 알고리즘을 만들 수 있다. 자세한 내용은 이후 포스트에서 서술하겠다.

<h3>7</h3>
![alt text]({{"/assets/images/7.PNG" | relative_url}})  

Landmark 좌표를 알고 있으므로, 약간의 padding을 더해서 2차원 슬라이싱을 통해 특정 얼굴 영역을 쉽게 Cropping할 수 있다.
Face와 Eye모두 파이썬 클래스를 통해 객체화했고, 각 객체는 상응하는 프레임 내의 Bounding box의 parameter(X, Y, W, H)로 구성되도록 설계했다.   

X, Y는 Boundingbox의 좌측 상단의 픽셀 좌표(Resizing 이전 기준), 그리고 W, H는 Bounding box의 가로세로 길이에 해당된다. 

<h3>후기 및 보완점</h3>
해당 스크립트를 이용해 약 2300장 정도의 샘플을 추출해봤는데, 시간이 오래 걸리는 건 둘째치고 힘들다. 나중에는 PoG 1개에 프레임 1를 매핑하는 식이 아니라 2~3초짜리 영상을 clipping해서 속한 모든 프레임들에 한꺼번에 추출 시퀀스를 적용시키는 것이 훨씬 나을 것이다. 
