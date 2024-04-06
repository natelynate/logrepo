---
layout: post
title:  "Dlib의 face landmark detection을 이용한 샘플 추출 파이프라인"
date:   2024-04-04 01:15:16 +0900
categories: projects
tags: gamcheugi computervision
---

CNN 모델 기반 Appearance-Based 시선 추적 모델을 테스트해보기 위해서 가장 간단한 형태의 모델을 사용한 논문을 참고하였다. 
<a href="https://www.frontiersin.org/articles/10.3389/frai.2021.796825/full">Link to the Paper</a>
![alt text]({{"/assets/images/2024-04-04-CNN_test_ModelStructure.png" | relative_url}})

굉장히 간단한 구조인데, 두 개의 인풋 채널을 활용한 CNN모델이고, 사용자의 얼굴, 그리고 dlib을 통해 추출한 얼굴의 landmark coordinates를 받아 최종적으로 화면의 x,y의 좌표로 매핑하는 구조다. 

이때 CNN 모델은 VGG16 pre-trained model의 weights를 활용한다. 

레이어가 많지도 않고, epoch를 10 epoch밖에 안 돌렸다는 점, 그리고 모델 구조를 굉장히 상세하게 적어줘서 (다른 논문들은 꽤나 함축적으로 적어놔서 보고 구현하는, 경험치가 부족한 사용자 입장에서 좀 난감한 경우가 많았다) 이후 모델의 성능과 비교대조할 수 있는 베이스라인 모델로서 알맞은 모델이었다. 다른 논문들에 비해 입력 데이터가 지극히 단순해서 이게 되나? 같은 생각이 많은데 어쨌든 실제로 구현해봐야 알 듯하다. 인용 수가 적은 논문이라 그런지 깃헙에서 구현한 리포지터리를 찾을 수 없었다.

현재 논문 말고도 이후 논문 구현에서도 꾸준히 쓸 작정으로 웹캠으로 샘플을 추출하는 스크립트를 만들었다.  

1.  
![alt text]({{"/assets/images/2024-04-04-capture0.png" | relative_url}})
dlib의 frontal_face_detector()를 써서 현재 프레임에서 얼굴 영역을 찾는다.  
<br>
<br>
<br>
![alt text]({{"/assets/images/2024-04-04-capture1.PNG" | relative_url}})
dlib의 face_estimator()로 facial landmarks를 찾는다. 기본적으로 68개가 나오는데 이중 약 39개 정도면 모델 입력값으로 사용된다.
<br>
<br>
<br>
![alt text]({{"/assets/images/2024-04-04-capture2.PNG" | relative_url}})
CNN 모델에 입력값으로 넣기 위해 사이즈를 통일해줘야 한다. 논문에서는 (244, 244, 3)으로 맞췄기 때문에 Bounding Box 기준으로
Cropping 후 Opencv의 Resizing으로 이미지를 처리해준다. 

이때 Facial landmarks의 좌표들도 image Transformation에 맞춰서 계속해서 align해줬다. Image transformation을 먼저 하고 landmark를 찾는 것이 훨씬 less intensive했겠지만, cropping이나 resizing을 하고 face_estimator를 사용하니까 
검출률이 크게 떨어지는 문제가 있어 좌표를 찾아놓고 -> 좌표를 transformation에 맞춰 재조정하는, 살짝 번거로운 처리 과정을 거치게 되었다. 

그 과정에서 최대한 나름 모듈화를 적용해 체계적으로 샘플을 추출하는 시퀀스를 만들려고 했고, 그 과정에서 구조가 맘에 안 들어 리팩토링을 두 번 수행하느라 생각보다 시간이 더 걸렸다. 

