---
layout: post
title:  Evaluating Clustering Algorithms
date:   2024-02-16 19:15:16 +0900
categories: study
tags: theory machinelearning
---

Clustering algorithm을 적용하기 이전, 혹은 이후에 Clustering의 정확도를 label이나 다른 metadata를 참고하지 않고 정량적으로 평가하는 방법들에 관한 포스팅이다.

도서 `데이터 과학자와 데이터 엔지니어를 위한 인터뷰 문답집`에서 내용을 참고하였다.

군집화가 제대로 이루어졌는지 평가하기 위해서는, 실제 label과 비교했을 때 정오표를 보는 것이 가장 신뢰성 있는 평가 방법이다. 

하지만 label이 없는 unsupervised 방식의 경우에, 데이터셋 내에 몇 개의 군이 존재하는지도 모르는 상태에서 clustering이 제대로 이루어졌는지,

혹은 그 전에 해당 데이터셋에 clustering을 애초에 사용할 개연성이 있는지 사전에 평가하는 방법은 없을까? 

이 경우 특정 지표를 사용할 수 있다. 

1. Hopkins Statistics  
2. Elbow Method  
3. Gap Statistics  

그렇다면 Clustering된 결과들 중 어느 clustering 결과가 더 좋을 지 객관적인 지표로 비교하는 방법에는 다음이 있다:
 
1. 실루엣계수 Silhouette Coefficient
2. 평균제곱근표준편차 Root Mean Square Standard Deviation
3. R-Square


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
```

테스트로 정규분포에서 샘플링한 100개짜리 데이터셋 2개, 그리고 완전히 랜덤 추출한 동일 크기의 랜덤 데이터셋을 만들었다.

```python
# Generate Random Clustering Results

class_result = pd.DataFrame(columns=['class', 'x', 'y'])

for cls in range(2):
    mean = [1, 10][cls]
    std = [3, 5][cls]
    x_list = np.random.normal(mean, std, 100)
    y_list = np.random.normal(mean, std, 100)
    temp = pd.DataFrame({'class':[cls for _ in range(100)], 
                            'x': x_list, 
                            'y': y_list})
    class_result = pd.concat([class_result, temp], axis=0)
```


```python
# Generate Random points
rng_result = pd.DataFrame(columns=['class', 'x', 'y'])

for cls in range(2):
    x_list, y_list = [], []
    min_x, max_x= min(class_result['x']), max(class_result['x'])
    min_y, max_y= min(class_result['y']), max(class_result['y'])
    for _ in range(100):
        x_list.append(random.randrange(int(min_x), int(max_x)))
        y_list.append(random.randrange(int(min_y), int(max_y)))
    temp = pd.DataFrame({'class':[cls for _ in range(100)], 
                            'x': x_list, 
                            'y': y_list})
    rng_result = pd.concat([rng_result, temp], axis=0)
```

```python
plt.figure()
for cls in range(2):
    temp = class_result[class_result['class'] == cls]
    plt.scatter(temp['x'], temp['y'], label=str(cls), alpha=0.5)
plt.scatter(rng_result['x'], rng_result['y'], label='random', alpha=0.5)
plt.legend()
plt.show()
```
3개 데이터셋의 분포를 시각화하면 다음과 같다.  
![png](_posts\study\machinelearning\EvaluatingClusteringAlgorithm\image-1.png)


`라벨링이 없는` 클러스터링 결과를 가지고 분류 성능을 알려면:

1. 일단 일반화된 방법은 존재하지 않음. 군집의 종류나 특성에 따라 갈리기 때문. K평균클러스터링은 오차제곱합으로 평가할 수 있지만, 밀도에 기반한 군집의 경우 분포가 구형이 아닐 수도 있기에 곤란함.

2. 일단 데이터 군집의 특성을 파악하는 것이 첫 번째 단계이다.

중심에 의해 정의되는 군집의 경우: 주로 spherical distribution pattern을 따르며, 포인트에서 중심까지의 거리가 다른 군집 유형 대비 짧다.  

밀도에 의해 정의되는 경우: 밀도에 기반한 군집 정의가 필요함.

연결에 의해 정의되는 데이터 군집: 


=========================================================================================================================
1. `클러스터링 경향성 측정`: 데이터 분포에 비임의성 군집 구조가 존재하는지 테스트하는 단계. 
아주 간단하게 얘기해서 분리 가능한 클러스터로 나뉠 수 있는지 가장 기본적인 사실을 검증하는 단계다. 
클러스터 개수 k를 증가시키면서 클러스터링 오차가 어떻게 변화하는지 관찰한다. 만약 완전 랜덤이라면 k의 증감에 따라 큰 변화폭이 없을 것이다.

혹은 Hopkins Statistics로 공간상 데이터의 랜덤성을 측정할 수 있다. 

![image.png](_posts\study\machinelearning\EvaluatingClusteringAlgorithm\image-3.png)

데이터셋에서 임의의 n개의 포인트를 샘플링한 후, 각 포인트($n_i$)마다 가장 가까운 이웃한 다른 포인트와의 거리를 $y_i$라고 하고, 
샘플이 취할 수 있는 값 내에서 아예 임의의 포인트를 똑같이 n개 랜덤생성한 후 똑같이 가장 가까운 이웃과의 거리를 $x_i$라고 하자. 

만약 ${\Sigma}y_i = {\Sigma}x_i$ 이면 Hopkins Statistics가 대략 0.5에 가까워진다.

${\Sigma}y_i = {\Sigma}x_i$ 라는 건 평균적으로 `랜덤하게 생성하나 직접 샘플링해서 뽑으나` 평균적인 거리관계가 차이가 없다는 것이다.

즉 0.5에 가까울수록 현재 데이터 분포에 클러스터가 있지 않고 완전 랜덤일 가능성이 높은 것.  

만약 ${\Sigma}y_i != {\Sigma}w_i$ 이면 $y_i > x_i$ 이거나 $y_i < x_i$ 두 경우가 있을 터인데,  
<br>
<br>
<br>
${\Sigma}y_i >> {\Sigma}x_i$ 의 경우, `랜덤생성한 포인트`는 실제 샘플링된 포인트와 비교했을 때 `밀도가 상대적으로 옅다`는 것이므로   
${\Sigma}x_i$의 영향력이 떨어지고, 홉킨스 통계량은 1에 가까워진다.

${\Sigma}y_i << {\Sigma}x_i$ 의 경우면 랜덤생성했을 때 오히려 더 밀도가 높다는 뜻인데... 이러면 홉킨스 통계량은 0에 가까워지고,
아마도 똑같이 클러스터 존재 가능성은 낮아질 것.

<font color='yellow'> 요약하면 홉킨스통계량은 0 ~ 0.5에 가까우면 클러스터가 없다는 것이고, 1에 가까울 수록 클러스터 존재 가능성이 올라간다. </font>
 


```python
# 홉킨스통계량을 위의 예시 데이터로 직접 계산했을 때

def get_euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_min_neighbour_dist(x1, y1, dataset):
    min_dist = 1e99
    for x2, y2 in zip(dataset['x'], dataset['y']):
        if x1==x2 and y1==y2: # 자기 자신과는 neighbour로 간주하지 않음
            continue
        min_dist = min(get_euclidean_dist(x1, y1, x2, y2), min_dist)
    return min_dist

def get_hopkins_stat(w, u):
    return (u / (u + w))

# n이 적으면 랜덤성에 의해 결과가 달라질 수 있으므로 n=100으로 테스트
# class 0,1이 혼재되어 있는 200개짜리 데이터셋에서 n=100으로 샘플링
sampled_points = class_result.sample(100)
sample_dist_sum = sum([get_min_neighbour_dist(point[1][1], point[1][2], class_result) for point in sampled_points.iterrows()])
print(f"sampled points nearest neighbour distance sum: {sample_dist_sum}")

# random하게 100개 포인트를 찍은 후 똑같이 distance sum을 구함
hopkins_rng = pd.DataFrame(columns=['class', 'x', 'y'])
for cls in range(2):
    x_list, y_list = [], []
    min_x, max_x= min(class_result['x']), max(class_result['x']) 
    min_y, max_y= min(class_result['y']), max(class_result['y'])
    for _ in range(100):
        x_list.append(random.randrange(int(min_x), int(max_x)))
        y_list.append(random.randrange(int(min_y), int(max_y)))
    temp = pd.DataFrame({'class':[cls for _ in range(100)], 
                            'x': x_list, 
                            'y': y_list})
    hopkins_rng = pd.concat([rng_result, temp], axis=0)

rng_dist_sum = sum([get_min_neighbour_dist(point[1][1], point[1][2], class_result) for point in hopkins_rng.iterrows()])
print(f"randomly assigned points nearest neighbour distance sum: {rng_dist_sum}")

hopkins = get_hopkins_stat(sample_dist_sum, rng_dist_sum)
print(f"Hopkins Statistics is: {hopkins}")
```

    sampled points nearest neighbour distance sum: 99.24930826313032
    randomly assigned points nearest neighbour distance sum: 545.3280787131861
    Hopkins Statistics is: 0.8460242163804345
    

Hopkins Statistics를 계산해봤을 때 0.85가 나옴. 1에 가까우므로 랜덤 생성한 포인트보다 실제 샘플링 포인트들의 nearest neighbour가 훨씬 가깝다는 것을

의미함. <font color='red'>즉 cluster가 있다고 보는 것이 합리적임`</font>. (실제로 있음)


```python
# 동일한 홉킨스 통계량 계산 (n=100)절차를 랜덤 생성된 데이터셋 rng_result(p=200)에 적용

# n=100으로 샘플링한 포인트
sampled_points = class_result.sample(100)
sample_dist_sum = sum([get_min_neighbour_dist(point[1][1], point[1][2], class_result) for point in rng_result.iterrows()])
print(f"sampled points nearest neighbour distance sum: {sample_dist_sum}")

# random하게 100개 포인트를 찍은 후 똑같이 distance sum을 구함
hopkins_rng = pd.DataFrame(columns=['class', 'x', 'y'])
for cls in range(2):
    x_list, y_list = [], []
    min_x, max_x= min(class_result['x']), max(class_result['x']) 
    min_y, max_y= min(class_result['y']), max(class_result['y'])
    for _ in range(100):
        x_list.append(random.randrange(int(min_x), int(max_x)))
        y_list.append(random.randrange(int(min_y), int(max_y)))
    temp = pd.DataFrame({'class':[cls for _ in range(100)], 
                            'x': x_list, 
                            'y': y_list})
    hopkins_rng = pd.concat([rng_result, temp], axis=0)

rng_dist_sum = sum([get_min_neighbour_dist(point[1][1], point[1][2], rng_result) for point in hopkins_rng.iterrows()])
print(f"randomly assigned points nearest neighbour distance sum: {rng_dist_sum}")
hopkins = get_hopkins_stat(sample_dist_sum, rng_dist_sum)
print(f"Hopkins Statistics is: {hopkins}")
```

    sampled points nearest neighbour distance sum: 355.1284701988156
    randomly assigned points nearest neighbour distance sum: 394.2840798353878
    Hopkins Statistics is: 0.5261242019731223
    

랜덤한 데이터셋에 홉킨스 통계치를 계산한 결과로 0.53이 나온다. 

즉 클러스터링을 적용하기 <font color=yellow> 이전에 </font> 클러스터의 존재 유무를 판별할 수 있다.

=========================================================================================================================
2. `클러스터 개수 k 측정`: 클러스터 존재유무를 파악한 후, 실제 군집 수를 찾은 후 이에 기반하여 클러스터링 결과의 질을 판별할 수 있다.
군집 수 결정 방법에는 Elbow Method나 Gap Statistic 방법이 있다. [명시되진 않았지만 클러스터링 알고리즘을 실적용한다는 것이 가정된다]

2-1. Elbow Method는 휴리스틱 서치 방법으로 k를 늘려나가면서 지표의 상승/하강 경향을 보고 최적의 k에서 타협하는 기법이다. 

2-2. Gap Statistics은 클러스터 내 데이터 산포(dispersion)를 Null distribution에 비교하는 방식으로, gap statistics가 최대화되거나 혹은 특정 threshold를 넘는 k를 선택한다.

=========================================================================================================================
여러가지 클러스터링 결과들 간의 품질을 비교하기 위해서는 다른 지표를 동원해야 한다. 

1. 실루엣계수 Silhouette Coefficient
2. 평균제곱근표준편차 Root Mean Square Standard Deviation
3. R-Square
4. Hubert ${\Gamma}$ 통계량


```python
# 이전에 만들었던 클래스 2개짜리 샘플 데이터셋 class_result에 Sklearn K-Means, GMM을 사용한 결과를 4개의 지표를 통해 비교해보겠다. 

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X = np.array([[tuple[1], tuple[2]] for tuple in class_result.itertuples(index=False)]) # list[[x,y], [x,y]...] 형식으로 변경

gmm = GaussianMixture(n_components=2, 
                      covariance_type='spherical',
                      random_state=100)
kmeans = KMeans(n_clusters=2,
                random_state=100)
gmm.fit(X)
kmeans.fit(X)

gmm_labels = gmm.predict(X)
kmeans_labels = kmeans.predict(X)

# 분류결과 임시 저장
class_result_labeled = class_result.copy()
class_result_labeled['gmm_pred'] = gmm_labels
class_result_labeled['kmeans_pred'] = kmeans_labels
```


```python
class_result_labeled.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>x</th>
      <th>y</th>
      <th>gmm_pred</th>
      <th>kmeans_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-4.251536</td>
      <td>-4.397753</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.537998</td>
      <td>-2.876279</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-1.454530</td>
      <td>6.065033</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2.349574</td>
      <td>5.091523</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1.365169</td>
      <td>0.960030</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.metrics import accuracy_score

correct = 0
for a, b in zip(class_result_labeled['gmm_pred'], class_result_labeled['class']):
    if a == b:
        correct+=1
print(correct)

correct = 0
for a, b in zip(class_result_labeled['kmeans_pred'], class_result_labeled['class']):
    if a == b:
        correct+=1
print(correct)
```

    183
    182
    

### Silhouette Coefficient (1/4)

실루엣 계수는 개별 포인트 p에 대해서 계산된다. 

![Alt text](_posts\study\machinelearning\EvaluatingClusteringAlgorithm\image-2.png)



<font color=green> a = 포인트 p가 속한 군집의 다른 포인트 p' 사이의 평균 거리

b = 포인트 p와 다른 군집의 포인트들 사이의 평균 거리(클러스터가 n개 있으면 가장 가까운 타 클러스터만 계산) </font> 


즉 a는 intra-cluster density를 나타내고, b는 군집 간의 떨어진 정도를 반영한다고 보면 된다.

즉 a와 b는 반비례 관계이고, a가 낮고 b가 높은 경우가 군집 품질이 좋다고 할 수 있다.

모든 포인트 p에 대해 실루엣계수를 계산한 후 그 평균값으로 전체 클러스터링 결과를 측정할 수 있다. 

-> 책에서는 b를 계산할 때 "최소 평균 거리"라고 해서 살짝 헷갈림. 괄호로 보충설명해놓긴 했지만 그냥 가장 가까운 클러스터 1개 간에 포인트들 간 평균거리

---

a <<< b인 경우, 분모는 $max(a, b)$ 이기 때문에 자연스럽게 b가 된다.   
그러면 b와 a가 차이가 많이 날 경우 (즉 군집 내부적으로 밀집해 있으면서 군집들끼리는 잘 떨어져 있는 경 우)  
silhouette score는 1에 가까워진다.  

반대로 b와 a가 차이가 많이 나지 않을 경우, 대신 0에 가까워진다.   

a >>> b인 경우, 분모는 a가 되고, 실루엣계수는 음수가 된다.  
a와 b가 비슷할 경우 음수지만 0에 가까워지고 (클러스터 없음)  
a가 b보다 훨씬 클 경우 -1에 가까워진다. 

즉 실루엣스코어는 -1 ~ 1 사이의 범위이고 1에 가까울 수록 명확한 클러스터링이 된 상태이다. 


```python
def get_silhouette_coef(x1:int, y1:int, label:int, dataset:pd.DataFrame):
    
    n1 = dataset[dataset['class'] == label].shape[0]
    # Get intra-cluster-distance
    a = sum([get_euclidean_dist(x1, y1, tuple[1], tuple[2]) for tuple in dataset.itertuples(index=False) if tuple[0] == label]) / n1
    
    # Get inter-cluster-distance
    n2 = dataset[dataset['class'] != label].shape[0] 
    b = sum(get_euclidean_dist(x1, y1, tuple[1], tuple[2]) for tuple in dataset.itertuples(index=False) if tuple[0] != label) / n2

    return (b-a) / max(a, b)

gmm_result = pd.DataFrame({'class':class_result_labeled['gmm_pred'],
                           'x':class_result_labeled['x'],
                           'y':class_result_labeled['y'],
                           })
kmeans_result = pd.DataFrame({'class':class_result_labeled['kmeans_pred'],
                           'x':class_result_labeled['x'],
                           'y':class_result_labeled['y'],
                           })

# GMM 결과에 대해 silhouette coef 계산
gmm_result['silhouette_coef'] = gmm_result.apply(lambda row: get_silhouette_coef(row['x'], row['y'], row['class'], gmm_result), axis=1)

# Kmeans 결과에 대해 solhouette coef 계산
kmeans_result['silhouette_coef'] = kmeans_result.apply(lambda row: get_silhouette_coef(row['x'], row['y'], row['class'], kmeans_result), axis=1)
```


```python
# 평균 silhouette score로 비교
kmeans_result['silhouette_coef'].mean(), gmm_result['silhouette_coef'].mean()
```




    (0.5065656873856117, 0.4986042521926654)




```python
# sklearn에 수록된 메서드로 계산해봤을때: two decimal places까지는 동일함. 근데 200개 중에 180개 맞춘 클러스터링인데 0.5밖에 안되네
# 아마 두 클러스터 거리가 되게 가까워서 그런 듯?
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.fit_predict(X)), silhouette_score(X, gmm.fit_predict(X))
```
    (0.5014271330076178, 0.49349007551874735)

============================================================================================================================================

평균제곱근 표준편차 Root Mean Squared Standard Deviation RMSSTD (2/4)

클러스터링 결과의 밀집성을 평가하는데 사용된다. 

$$ \sqrt{\frac{{\Sigma}_i{\Sigma}_{x \in C_i} \parallel (x-c_i) \parallel^2}{{\Sigma_{j=i...p}} (n_{ij} - 1)}}$$

Where 

$c_i$ = `i번째 Cluster $C_i$의 중심점  `

$p$ = `독립변수 개수` (예시에서는 2차원이므로 2개)



공식을 해석하면 

"모든 개별 클러스터 내에서 자기 군집의 Centroid 간 거리의 평균제곱합의 합을 전체 데이터 공간에서 수행한 후, 그 결과를 (포인트 개수 - 클러스터 개수)로 
나눠서 일종의 평균제곱합들의 평균을 내는 것과 같다. 

즉 정규화가 반영된 표준편차라고 보면 된다.

RMSSTD는 당연히 낮을수록 좋다 (Centroid 주변에 밀집해 있을 수록 명확한 클러스터이므로)

___

일단 RMSSTD를 단독으로 클러스터링 평가에 쓰는 건 아주 흔하진 않은 듯 하다 (검색결과가 현저히 적음)
사용할 수는 있다-> 정도로 알아두자.

(1,1)보다 (1,1,1)이 거리가 커지므로 데이터 차원이 커질수록 포인트간 거리가 커지므로 자연히 오차제곱합도 커진다.
따라서 상수 P를 분모에 넣어서 나눠줌으로써 차원에 따라 결과가 커지는 걸 막는 정규화를 수행한다. (3차원이면 P=3이다)


참고문헌: 
https://www.researchgate.net/publication/267369576_Comparison_of_Clustering_Techniques_for_Cluster_Analysis

===============================================================================================================
R Square(3/4)

군집 사이의 차이 정도를 측정할 수 있다. 

공식을 보면 (전체오차제곱합 - 클러스터링 후 오차제곱합합) / 전체오차제곱합 의 형태이므로, 

클러스터링이 전체오차제곱합을 얼마나 개선했는지 나타내는 지표이다. 

즉 클러스터링이 효과가 없으면 이전이나 이후나 똑같으므로 0이 나오는 구조이고, 클러스터링 후 오차제곱합이 줄어들수록 1에 가깝게 결과가 나오는 구조다.

===============================================================================================================


