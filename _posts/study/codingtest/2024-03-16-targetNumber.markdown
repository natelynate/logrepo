---
layout: post
title:  "[Programmers Lv.2] 타겟 넘버"
date:   2024-03-16 19:15:16 +0900
categories: study
tags: codingtest dfs
---
굉장히 간단하고 모범적인 재귀식 dfs로 풀 수 있었다. adjacency list를 사용하지 않고 매번 2개의 분기 중 하나로만 나뉘는 경우.

```python
# dfs
def solution(numbers, target):
    answer = 0
    def dfs(idx, result):
        nonlocal answer 
        if idx == n:
            if result == target:
                answer += 1
            return
        else:
            dfs(idx+1, result+numbers[idx])
            dfs(idx+1, result-numbers[idx])
    n = len(numbers)
    dfs(0, 0)
    return answer
```