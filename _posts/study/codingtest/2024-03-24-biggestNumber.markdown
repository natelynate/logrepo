---
layout: post
title:  "[Programmers Lv.2] 가장 큰 수"
date:   2024-03-24 19:15:16 +0900
categories: study
tags: codingtest programmers sorting
---

https://school.programmers.co.kr/learn/courses/30/lessons/42746

굉장히 여러가지 방식으로 계속 시도했었고 거의 근접하기도 했었던 것 같은데 결국 최종 해답에 도달하지 못했다.
전부다 4자리수로 맞춰야 한다는 건 이해했는데 000으로 buffer를 넣을 생각만 했지.. 이러면 10이랑 1이랑 동일해지는 문제가 생겨서.
해답은 0으로 buffer를 넣는게 아니라 원 문자열을 반복하는 것이다. 

3 vs 34 -> 3333 vs 3434 이런 식으로.

또한 sort의 key에 lambda에 변환식을 작성해주면 리스트의 원소들이 해당 변환식으로 변환된 기준으로 sorting되지만 원소 자체가 변하지는 않으므로 굳이 인덱스를 keeping했다가 재구성하는 데 쓸데없이 힘을 뺄 필요가 없다. 

```python
def solution(numbers):
    numbers = list(map(str, numbers))
    
    numbers.sort(key=lambda x: (x*4)[:4], reverse=True)
    
    if numbers[0][0] == '0': # if the sum is 0, specify an exception to return 0 instead of 0000
        return '0'
    else:
        return ''.join(numbers)
```