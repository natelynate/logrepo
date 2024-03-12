---
layout: post
title:  다익스트라, 플로이드-워셜 알고리즘(Dijkstra, Floyd-Warshall, Bellman Ford)
date:   2024-01-12 19:15:16 +0900
categories: study
tags: codingtest, theory, algorithm
---

최단거리를 구하기 위해 활용될 수 있는 알고리즘 3개를 설명한다. 
1. 다익스트라 
2. 플로이드-워셜
3. 벨만 포드

<h3>다익스트라 알고리즘</h3>

`그래프 내 임의의 노드에서 다른 모든 노드들까지의 최단 경로`를 구한다.    
0. 다른 모든 노드들까지 가는 비용을 저장하는 n길이의 '최단 거리 array'를 생성한다(n = number of nodes).   
1. 현재 위치한 노드의 인접 노드 중 방문하지 않은 노드를 구별하고, 방문하지 않은 노드 중 거리가 가장 짧은 노드를 선택한다.   
2. 그 노드를 방문 처리한다.  
3. 해당 노드를 거쳐 다른 노드로 넘어가는 간선 비용(가중치)을 계산한다.  
4. 새로 계산한 경로의 비용이 기존에 '최단 거리 array'에 등록된 비용보다 낮다면 최단 거리 테이블을 업데이트하고 해당 인접 노드를 queue에 삽입한다. 
5. queue가 소진될때까지 반복하면 모든 노드들간의 최소 거리가 저장된 array를 얻을 수 있다.

heapq를 쓰면 손쉽게 사용할 수 있다.

대략적인 코드 예시는 다음과 같다. 
```python
import heapq

n = 10
graph = [[]] # node info is a tuple of (node_num, cost)
min_dist = [int(1e9) for _ in range(n+1)] # array to save minimum distance to each node from the starting node
start_node = 0

def dijkstra(start):
    queue = list()
    heapq.heappush(queue, (0, start)) # heappush the start node info. Node info is a tuple of (distance, node_num) 
    min_dist[start] = 0 

    while queue:
        dist, node_idx = heapq.heappop(queue)
        if min_dist[node_idx] < dist: # if the node is already visited, skip the instance
            continue
        for node in graph[node_idx]: # iterate the adjacent nodes connected to the current node
            cost = dist + node[1]

            if cost < min_dist[node[0]]: # if the new cost is smaller than the one saved at the min_dist array
                min_dist[node[0]] = cost
                heapq.heappush(q, (cost, node[0])) # push the node into the heapq
```

다익스트라의 단점:

기본형의 경우 $O(V^2)$이지만 우선순위 queue를 사용할 경우 O((V+E)logV), v=vertex개수, E=한 vertex의 주변 노드로 줄일 수 있다.  

다익스트라의 단점이라면 음수 간선이 존재하는 그래프일 경우 사용할 수 없다는 것이다. 정확히는 cycle이 존재하면서 해당 cycle에

음수 간선이 존재할 경우 무한루프에 빠질 수 있다. 매번 heapq에서 v라는 노드를 뽑았을 때, 해당 v와 연결된 다른 노드들 e로 이동하는 

비용을 새로이 계산한다. v와 e에서 이어지는 간선 중 음수가 있으면, 최소 비용을 선택한다는 로직 상 항상 해당 음수 간선이 선택되게 되고,

같은 정점만 반복해서 뽑게 되면서 무한루프가 발생한다. cycle이 없으면 비록 느리지만 동작은 하나, 기왕이면 벨만포드를 쓰자.



<h3>플로이드 워셜 알고리즘</h3>

다익스트라와 다르게, n개의 노드가 있을 때 `모든 노드에서부터 다른 모든 노드들간의 최단거리`를 예외없이 모두 구하는 알고리즘이다. 

노드 a에서 c로 가는 최단거리가 graph[a][c] 라고 했을 때, a와 c 사이에 존재하는 모든 노드 b에 대해서 min(a, b) + min(b, c)를 합한 것이 곧 
a -> c 의 최소 거리가 된다는 논리이다. 가능한 모든 연결의 경우의 수를 탐색하기 때문에 자연스럽게 시간 복잡도 $O(N^3)$ 이다.
```python
# Floyd-Warshall Algorithm

INF = int(1e9)
n = int(input()) # 노드 숫자
m = int(input()) # 간선 숫자

# 2차원 리스트 생성 및 초기값 부여
graph = [[INF] * (n + 1) for _ in range(n + 1)]

# 자기 자신에서 자기 자신으로가는 비용은 0으로 초기화
for a in range(n + 1):
    for b in range(m + 1):
        if a == b:
            graph[a][b] = 0

# 각 간선에 대한 정보를 입력받아 그 값으로 초기화
for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a][b] = c

# 점화식에 따라 플로이드 워셜 알고리즘 수행
for k in range(1, n+1):
    for a in range(1, n+1):
        for b in range(1, n+1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# 수행 결과 출력
for a  in range(1, n + 1):
    for b in range(1, n + 1):
        if graph[a][b] == INF:
            print("INFINITY", end="")
        else: 
            print(graph[a][b], end=" ")
```

<h3>벨만포드 알고리즘</h3>

Weighted-Directed Graph에서 임의의 노드에서 다른 모든 노드들 간의 최단거리를 찾는다. 이 점에서 다익스트라와 목적을 공유한다.
그러나 $O(VE)$로 다익스트라보다 다소 느린 대신 음수 간선이 있을 때도 사용할 수 있다.

다익스트라가 `"미방문 노드 중 최단 거리가 가장 짧은 노드"`를 우선적으로 탐색하는데 비해   
벨만 포드에서는 `매번 모든 간선을` 확인한다. 

전 과정은 다음과 같다.

0. 시작노드와 최소거리 array를 설정한다.  
1. 이하의 과정을 N-1번 반복한다.  
    1.1 전체 간선 E개를 하나씩 확인한다.
    1.2 각 간선을 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리를 갱신한다. 이때 출발노드가 방문한 적 없는 노드(INF)면 값을 업데이트하지 않는다.

2. 만약 음수 간선 순환이 발생하는지 체크하고 싶다면 3번의 과정을 한 번 더 수행한다.
    → 이 때 최단 거리 테이블이 갱신된다면 음수 간선 순환이 존재하는 것이다.

```python
# 1번 노드에서 시작한다고 가정했을 때

N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        edges.append((A, B, C))
    dist_array = [1e99 for _ in range(N+1)] # 최소 거리 array
    dist_array[1] = 0 # 출발 노드에서 출발 노드와의 거리는 0

def bellman(N, M, dist_array):
    for i in range(1, N+1): # 모든 노드에 
        for edge in edges:
            A, B, C = edge 
            # B까지 가는 기존 비용 vs (A->B 경로를 통해 B로 가는 비용 비교)
            if dist_array[A] != 1e99 and dist_array[B] > dist_array[A] + C:
                dist_array[B] = dist_array[A] + C
                if i == N: # 만약 마지막 노드를 점검 중임에도 값 갱신이 발생한다면 음수 cycle이 존재하므로 최소결과를 찾을 수 없음. 
                    return -1
    return dist_array 
```