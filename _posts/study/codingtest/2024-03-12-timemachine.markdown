---
layout: post
title:  [Baekjun] 타임머신
date:   2024-03-12 19:15:16 +0900
categories: study
tags: codingtest bellmanford
---

```python
# https://www.acmicpc.net/problem/11657
# 최단거리 -> 벨만포드 알고리즘 
# 음수간선이 존재하므로 다익스트라 사용불가. 문제 내에서 "사이클 여부"를 묻고 있으므로 사이클 검출이 가능한 벨만포드가 제일 유리함.
# 무한 값을 충분히 크게 잡아야 함 -> 1e9 에서 오류 뜸. 
# n=1, n=2 등 엣지케이스에서 답 출력 포맷을 잘 확인해야 함. 
# 음수사이클 판별일 때 N이 마지막 run인지 N-1이 마지막 run인지 잘 확인할 것..

import sys
import math

def bellman(N, dist_array):
    for i in range(1, N+1): # 모든 노드에 
        for edge in edges:
            A, B, C = edge 
            # B까지 가는 기존 비용 vs (A->B 경로를 통해 B로 가는 비용 비교)
            if dist_array[A] != math.inf and dist_array[B] > dist_array[A] + C:
                dist_array[B] = dist_array[A] + C
                if i == N: # 만약 마지막 노드를 점검 중임에도 값 갱신이 발생한다면 음수 cycle이 존재하므로 결과를 반환
                    return -1
    return dist_array 
                
if __name__ == '__main__':
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        edges.append((A, B, C))
    dist_array = [math.inf  for _ in range(N+1)] # 최소 거리 array
    dist_array[1] = 0 # 출발 노드에서 출발 노드와의 거리는 0

    answer = bellman(N, dist_array)

    if answer == -1:
        print(answer)
    else:
        if len(answer) == 2:
            if answer[-1] == math.inf:
                print(-1)
            else:
                print(answer[-1])
        else:
            for i in range(2, N+1):
                if answer[i] == math.inf:
                    print(-1)
                else:
                    print(answer[i])
```