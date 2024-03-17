---
layout: post
title:  "[Programmers Lv.3] 단어 변환"
date:   2024-03-17 19:15:16 +0900
categories: study
tags: codingtest dfs
---


```python
'''
단어와 단어 간 간선이 있다고 보면 된다.
간선이 있는 조건은 단어의 철자 중 다른 게 1개만 있을 경우 간선이 있다고 생각하면 된다. 
dfs는 임의의 노드에서 시작해서 target 노드까지 도달하는 최단거리를 찾는 셈이다. 최대거리는 전체 노드 개수와 같다. 
(전체 노드 개수 이상부터는 탐색할 필요 x)
여기서 최대거리를 잘못 계산해서 조금 헤맴. 
'''

def is_adjacent(word1, word2):
    # score가 len(word)-1이어야 word1과 word2가 변환 가능
    score = 0
    for i in range(len(word1)):
        if word1[i] == word2[i]:
            score += 1
    if score == len(word1)-1:
        return True
    return False

def solution(begin, target, words):
    def dfs(current_word, steps):
        nonlocal answer
        if current_word == target:
            answer = min(answer, steps)
            return
        if steps == len(current_word)+1:
            return
        # 종료조건 미충족시 인접 단어 탐색
        for word in words:
            if word == current_word:
                continue
            if is_adjacent(current_word, word): 
                dfs(word, steps+1)
    answer = 1e9  
    dfs(begin, 0)
    
    return answer if answer != 1e9 else 0

solution('hit', 'cog', ["hot", "dot", "dog", "lot", "log", "cog"])
```