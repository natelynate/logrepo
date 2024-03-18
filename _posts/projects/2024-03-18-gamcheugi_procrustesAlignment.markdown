---
layout: post
title:  "Procrustes Algorithm 프로크루스테스 알고리즘"
date:   2024-03-18 19:15:16 +0900
categories: projects
tags: gamcheugi computervision
---

### 이론
Procrustes Algorithm라는 polygon alignment 테크닉에 대해 간략하게 서술한다.

Procrustes Algorithm은 도형 집합이 있을 때, 각 도형들의 shape variation의 통계치를 측정하고, 기준이 되는 도형이 있다면 해당 기준 도형과 비교해서 집합의 shape variation이 최소로 수렴할 때까지 반복적으로 alignment를 반복하는 알고리즘이다. 

알고리즘에 다소 블랙유머적인 명칭이 붙었는데, procrustes(프로크루스테스)는 그리스 신화에서 테세우스에게 퇴치당하는 악랄한 강도의 이름이다. 이 강도는 행인을 납치해서 자기 집의 침대에다 묶어놓고 만약 침대보다 더 크면 삐져나온 신체 부위를 도끼로 잘라 살해하고, 침대보다 작으면 침대 사이즈에 맞을 때까지 사지를 밧줄로 당겨 살해하는 강박 증세를 가진 강도인데, 도형들을 늘리고 줄여서 기준 도형(전 이야기에서 침대의 역할을 하는)에 맞춘다는 점에서 어느 누군가가 프로크루스테스의 이야기를 떠올렸던 것 같다.

어쨌든 프로크루스테스 알고리즘은 다음과 같은 방식으로 진행된다.
[평균 도형 기준]
1. 기준 도형을 택한다. 기준은 (서로 유사한 도형들의 집합이라면) 0번째 도형이거나, 혹은 평균 도형을 택할 수도 있다.
2. 기준 도형 제외 나머지 도형들은 기준 도형과 최대한 유사하게 변환될 수 있는 scale, rotation, translation을 계산한다.
   Procrustes distance를 최소화하는 변환을 찾으면 된다. Procrustes distance라고 해서 뭔가 싶지만 그냥 각 포인트 별 유클리드 거리이다. 즉 두 shape가 완전히 똑같다면, Procrustes distance는 0이다.
3. 계산한 수치를 가지고 각 도형을 superimpose한 후 mean shape를 새로 계산한다.
4. 만약 mean shape와 reference shape의 procrustes distance가 특정 threshold 이상이라면, 2번부터 다시 반복한다.

위 과정을 대략 shape가 수렴할 때까지 반복하면 도형들이 최대한 유사하게 align된다.

한 가지 주의사항으로는 point단위로 수행되기 때문에 도형들의 포인트 개수와 relative ordering에 신경써야 한다는 점이 있다.


### 실전
dlib의 find_affine_transform()으로 쉽게 수행할 수 있다.





