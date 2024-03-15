---
layout: post
title:  "[논문 리뷰] Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark"
date:   2024-02-05 19:15:16 +0900
categories: projects
tags: gamcheugi paper
---

본 포스트는 Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark(2021) 논문의 내용을 간략히 정리한 내용입니다. 

실시간 Eye tracking에는 크게 3갈래의 접근 방식으로 나뉩니다. 

1. <b><font color=red>3D Eye 모델 기반 방법</font></b>은 안구의 3차원 모델을 사용자 맞춤 형태로 구축하는 방식을 사용합니다. 이 경우 나름 의미 있는 수준의 정확도를 보여주지만, 적외선 카메라 등 전용 측정 장비를 요구합니다.

2. <b><font color=red>2D Regression</font></b> 방식은 3D Eye 모델과 유사한 전용 장비를 요구하면서, 안구의 동공 중심점이나 glint 등의 feature를 가지고 화면의 응시 지점(Point of Gaze, PoG)를 즉히 도출하는 회귀 모델을 구축하는 방식입니다. 결과값이 바로 PoG 이기 때문에 시선 벡터를 도출하고, 해당 시선 벡터(Gaze Vector)에 상응하는 화면 지점을 계산하는 별도의 방법이 필요 없다는 장점이 있습니다. 

3. <b><font color=red>Appearance-based Model</font></b>은 상용 웹캠 장비로 촬영한 영상을 기반으로 Gaze Vector를 추출한 후 이를 응시 지점과 매핑하는 방식을 취합니다. 매핑 함수는 본질적으로 Regression함수이고, 훈련시키기 위해 많은 샘플이 필요하다는 것과, 고차원 이미지 데이터를 저차원으로 매핑하는 회귀 함수의 성능이 1,2번에 비해 좋지 않다는 단점이 있습니다. 

![alt text]({{"/assets/images/2024-02-15-gamcheugi_PaperReview1.PNG" | relative_url}})

3번의 경우, Feature Extraction을 통해 확보한 Feature를 가지고 별도의 Regression 모델을 훈련시키는 것보다는 최근 딥러닝을 이용해 PoG나 혹은 시선 벡터를 바로 도출해내는 방식이 시도되고 있고, 좋은 성과를 거두고 있습니다. 

![alt text]({{"/assets/images/2024-02-15-gamcheugi_PaperReview2.PNG" | relative_url}})

CNN 모델이 처음 사용되었을 때, 대부분의 기존 appearance-based models보다 뛰어난 성능을 보였습니다. (Zhang et al)

#### Deep Feature From Appearance

기본적으로 CNN모델이든 Regression모델이든 눈의 형태적 변화 (동공의 움직임, 눈꺼풀의 모양)와 시선 움직임과 연관이 되어 있다는 사실에서부터 출발합니다.

CNN 모델의 경우 고차원 이미지 데이터에서 자동으로 feature를 추출합니다. 다른 딥러닝 모델과 비슷하게, 딥러닝 모델의 깊이와 복잡함에 비례하여 추출된 feature가 제공하는 정보의 양이 많아집니다.

또한 최초의 CNN 모델에서는 단일한 눈 image를 기반으로 수행되었으나, left eye와 right eye에 각각 feature extraction을 수행한 후 이를 concatenate하여 다음 Fully-Connected 레이어의 인풋으로 활용하는 경우도 제안되었습니다. 

![alt text]({{"/assets/images/2024-02-15-gamcheugi_PaperReview3.PNG" | relative_url}})

그림에서는 총 4개의 source로부터 추출한 deep feature를 입력으로 활용하는 경우의 예시입니다. 좌안 + 우안 + 전체 얼굴 이미지 + 얼굴이 위치한 Grid 좌표로 중복을 감수하더라도 (좌안+우안은 전체 얼굴 이미지에 이미 포함됨) 들어가는 경우인데, 일종의 랜드마크 격인 논문인 "Gaze Tracking for Everyone" 에서 사용된 조합이기도 합니다. 이는 눈 이미지만 단독으로 사용한 경우보다 좋은 결과를 보였습니다. 

얼굴 이미지에서 추출되는 deep feature의 예시로는 head pose (roll, pitch, yaw 등) 이나 facial landmark 등이 있으나, `오히려 직접 추출한 feature를 사용하는 것은 얼굴 이미지 전체를 CNN의 자동 feature 추출을 이용하는 것보다 못하다는` 결과도 나왔습니다. 

#### Feature from video
static image 뿐만 아니라 영상의 temporal information을 활용한 연구도 있습니다. 이 경우 흔히 RNN, LSTM이 사용됩니다. 주로 각 프레임에서 CNN을 적용해 feature를 추출한 후 그 feature들을 RNN에 넣어서 자연스럽게 temporal information이 반영되도록 하는 구조를 취합니다.

fixation, saccade, smooth pursuits과 같은 동공 움직임 패턴을 반영하여 정확도를 올리기 위한 시도도 있었습니다.

###### 추가 사항
fixation = 특정한 지점에서 고정
saccade = 짧은 시간 동안 특정한 곳으로 이동
smooth pursuits = (특히 움직이는 물체를 보기 위해) 끊기는 구간 없이 부드럽게 시선이 따라가는 움직임

이 있습니다. 이중 smooth pursuits는 실제 움직이는 물체가 제공되지 않았을 경우 사람이 해당 움직임을 자체적으로 재현하기 힘들어 합니다. 즉 일반적인 경우에서 시선 움직임은 연속적이지 않고, 다소 이산적인 형태로 '이리저리 튀는' 움직임을 보이게 되고, 이를 saccade라고 합니다. 







