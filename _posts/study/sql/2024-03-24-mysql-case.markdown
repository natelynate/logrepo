---
layout: post
title:  "CASE Syntax in MySQL"
date:   2024-03-24 19:15:16 +0900
categories: study
tags: sql
---

CASE는 일종의 if문과 유사한 로직을 사용할 수 있게 해준다.

기본적으로 CASE로 시작하고 END로 블록을 끝낸다. 

내부의 동작 논리는 WHEN {CONDITION} THEN {VALUE1} ELSE {VALUE2} 형식으로 기술할 수 있다. 파이썬에서 

```python
[1 if i == 1 else 0] 
```

과 같은 식으로 if-else문을 1줄에 적는 syntax와 유사하다. 

비슷하게 IF(condition, value_if_true, value_if_false) 와 같은 내장함수도 존재한다. 

간단한 조건문을 쓸 때의  간결함은 IF()함수가 유리하고, 만약 nested되기 시작하거나 복잡해지는 경우 가독성의 측면에서 CASE가 나을 때가 있다.

