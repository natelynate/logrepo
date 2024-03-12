---
layout: post
title:  Naive Bayes Classifier Theory
date:   2024-02-14 19:15:16 +0900
categories: study
tags: theory machinelearning
---

사실 Naive Bayes는 Iterative parameter Optimization을 하지 않기 때문에 다른 친숙한 ML 알고리즘이랑은 양상이 다르고 


광의의 의미에서만 머신러닝이라고 할 수 있다. 


지정된 샘플이 특정한 클래스에 속할 확률인 $P(y_i \vert \theta)$ 를 베이지안 통계를 통해 예측함으로써 진행된다.


$x_i = \theta_i$ 


어떤 샘플이 ${\theta_1, \theta_2, \theta_3,} $ 이렇게 있었을 때 모든 $\theta$를 독립이라고 간주함 (그래서 naive임)


독립으로 간주하게 되면 계산이 무척 쉬워지는게


$P(y_i \vert \theta_1, \theta_2, \theta_3)$에 베이즈 정리를 사용하면 


$P(y\vert \theta) = \frac{P(\theta\vert y)}{P(\theta)}$


$P(\theta)$는 샘플의 사전확률에 해당되는데 여기서도 각 샘플의 사전확률은 특정 샘플 x와 임의의 클래스 y에 대해 확률은 동일하다고 가정하면   
전체 계산 결과에 영향을 주지 않으므로 빼버려도 그만이다. 


그렇다면 


$P(y\vert \theta) = P(\theta\vert y) = P(\theta_1\vert y) * P(\theta_2\vert y) * P(\theta_3\vert y)$


Naive Bayes는 Supervised 이므로 라벨링된 샘플데이터셋이 있으므로, 샘플 데이터셋을 기계적으로 계산해서 $P(\theta_i\vert y)$를 각각 계산하고


이후에 새로운 라벨링을 해야 하는 상황이 생기면, 즉 $P(y_i\vert x)$ 를 계산해야 할 상황이 생기면 


다시 $\Pi{p(\theta_i\vert y_i)}$ 를 개별 라벨 $y_i$마다 진행한 후 그 결과를 $Result_i$ 이라고 했을 떄


그 중 제일 높은 베이즈 확률을 보인 걸 반환해주면 된다. 


즉 $max(Result_1, Result2, Result3)$ 이다. 


이를 확률그래프 모형으로 표기하면:

![image.png]({{"/assets/images/2024-02-14-NaiveBayesClassifier_1.png/"|  relative_url}})

