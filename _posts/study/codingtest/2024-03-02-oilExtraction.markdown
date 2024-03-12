---
layout: post
title:  "[Programmers Lv.2] 석유 시추"
date:   2024-03-02 19:15:16 +0900
categories: study
tags: codingtest bfs exhaustiveSearch
---

programmers level 2  
https://school.programmers.co.kr/learn/courses/30/lessons/250136  
Level 2
BFS에서 첫 원소를 queue에 넣을 때 visited를 업데이트해주는 걸 깜빡해서 자꾸 숫자가 +1인 상태였었음

```python
from collections import deque

def solution(land):
    def bfs(col, row, deposit_id):
        deposit_amount = 0
        queue = deque([(col, row)])
        visited[col][row] = True
        while queue:
            col, row = queue.popleft()
            deposit_amount += 1
            land[col][row] = deposit_id + 1
            for i in range(4):
                n_col = col + dy[i]
                n_row = row + dx[i]
                if not (0 <= n_col < num_col) or not (0 <= n_row < num_row):
                    continue
                else:
                    if land[n_col][n_row] != 0 and not visited[n_col][n_row]:
                        visited[n_col][n_row] = True    
                        queue.append((n_col, n_row))
        return deposit_id+1, deposit_amount
    
    num_col = len(land) # 열 길이 
    num_row = len(land[0]) # 행 길이
    visited = [[False] * num_row for _ in range(num_col)]
    deposit_id = 2
    deposit_dict = {}
    dy, dx = [-1, 1, 0, 0], [0, 0, -1, 1]
    for col in range(num_col):
        for row in range(num_row):
            if land[col][row] == 1 and not visited[col][row]:
                deposit_id, deposit_amount = bfs(col, row, deposit_id)
                deposit_dict[deposit_id] = deposit_amount
    max_extraction = 0
    for y in range(num_row):
        reservoir = 0
        ids = []
        for x in range(num_col):
            if land[x][y] == 0:
                continue
            else:
                if land[x][y] not in ids:
                    ids.append(land[x][y])
                    reservoir += deposit_dict[land[x][y]]
            max_extraction = max(max_extraction, reservoir)
    return max_extraction  
        
if __name__ == "__main__":
    case2 =[[1, 0, 1, 0, 1, 1], 
            [1, 0, 1, 0, 0, 0], 
            [1, 0, 1, 0, 0, 1], 
            [1, 0, 0, 1, 0, 0], 
            [1, 0, 0, 1, 0, 1], 
            [1, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1]]
    print(solution(case2))
```
