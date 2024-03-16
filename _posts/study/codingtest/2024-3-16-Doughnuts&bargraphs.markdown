---
layout: post
title:  "[Programmers Lv.2] 도넛과 막대 그래프"
date:   2024-03-15 19:15:16 +0900
categories: study
tags: codingtest 
---
문제 링크: https://school.programmers.co.kr/learn/courses/30/lessons/258711

카카오 공채 문제 특: ~~개같이 어려움~~

레벨 2 치고 굉장히 어렵다고 생각했다. 일단 어디서부터 시작해야 할 지 전혀 감이 오지 않았던 것 같다. 

막대그래프, 도넛그래프, 8자 그래프 등 굉장히 위협적인 신개념들이 나와서 사고를 마비시키는데, 차근차근 무엇을 모르는지, 그리고 주어진 정보를 잘 정리해서 패턴을 찾는 것부터 시작한다. 

핵심 문제는 다음과 같다:  

    `1. 새롭게 추가되는 정점은 몇 번 정점인가?`

    `2. 각 그래프를 어떻게 구분할 수 있는가?`

일단 새롭게 추가되는 정점은 무조건 "각 그래프의 임의의 노드로 1개의 간선으로 연결된다" 라고 적혀있다. 이 말인즉슨 새롭게 추가되는 정점은 무조건 밖으로 나가는 간선만 있고, 들어오는 간선은 0이다. 그 외에 모든 노드들은 어쨌거나 최소 1개의 indegree를 가질 수 밖에 없다. 
즉 간선 리스트 edges를 보면서 "나가는 간선만 있는" 정점을 찾으면 그게 새롭게 추가되는 정점이고, 그 정점이 가지고 있는 간선 개수만큼 그래프가 주어진다는 걸 알 수 있다.

다음과 같은 과정으로 새롭게 정점을 찾았다.
adj_dict에 노드별로 인접노드를 리스트로 저장해서 이후 탐색을 위해 미리 준비했다. 

```python 
node_info = {}
adj_dict = {}
answer = [0, 0, 0, 0]
# 진입차수가 -2 이하인 노드 = 생성한 정점 
for edge in edges:
    if edge[0] not in node_info:
        node_info[edge[0]] = -1
        adj_dict[edge[0]] = [edge[1]]
    else: 
        node_info[edge[0]] -= 1
        adj_dict[edge[0]].append(edge[1])
        
    if edge[1] not in node_info:
        node_info[edge[1]] = 1
        adj_dict[edge[1]] = []
    else:
        node_info[edge[1]] += 1
                                        
for i in node_info.items():
    if i[1] <= -2:
        center_node = i[0]
        num_of_graphs = len(adj_dict[i[0]])
        break
```

다음은 각 그래프들의 특징이다. 서로 구분이 가는 특징을 찾을 수 있다. 

1. 인접노드가 없는 정점이 있다면 해당 그래프는 막대 그래프다.

2. 밖으로 나가는 간선이 2개가 있는 노드가 있다면 해당 그래프는 8자 그래프다. 

3. 다음 노드로 갔는데 만약 첫 번째 노드로 돌아온다면 해당 그래프는 도넛 그래프다. 

1,2,3번의 특징을 기준으로 어떤 유형의 그래프인지 알 수 있다.

DFS로 탐색하기로 해서 DFS 함수를 만들었다. DFS를 풀어본 지 좀 돼서 구조가 살짝 애매했는데 일종의 코드 머?슬메모리처럼 손 가는대로 적었더니 진짜 돼서 스스로도 조금 놀랐다. 

```python 
def determine_type(node_history:list):
    current_node = node_history[-1]
    next_nodes = adj_dict[current_node]
    if not next_nodes: # 인접노드가 없는 노드가 그래프 내에 존재한다면
        return 2 # 막대 그래프 +1
    elif len(next_nodes) == 2: # 나가는 간선이 2개 있는 정점이 있다면 8자그래프
        return 3 # 8자 그래프 +1
    elif len(next_nodes) == 1 and next_nodes[0] == node_history[0]:
        return 1 # 도넛 그래프
    else:
        node_history.append(next_nodes[0])
        type_code = determine_type(node_history)
        node_history.pop(-1)
        return type_code
```
해당 dfs를 새롭게 추가된 정점과 간선으로 이어진 노드별로 각각 실행해주면 유형별 그래프의 수를 파악할 수 있을 것이다. 

```python 
answer = [0, 0, 0, 0]
for adj_node in adj_dict[center_node]:
        type_code = determine_type([adj_node])
        answer[type_code] += 1
```

일부 테스트 케이스에서 에러가 나서 recursionlimit을 확장하고 다시 돌렸더니 통과됐다. 

전체 코드는 다음과 같다:
```python 
import sys

sys.setrecursionlimit(10000000)

def solution(edges):
    node_info = {}
    adj_dict = {}
    answer = [0, 0, 0, 0]
    # 진입차수가 -2 이하인 노드 = 생성한 정점 
    for edge in edges:
        if edge[0] not in node_info:
            node_info[edge[0]] = -1
            adj_dict[edge[0]] = [edge[1]]
        else: 
            node_info[edge[0]] -= 1
            adj_dict[edge[0]].append(edge[1])
            
        if edge[1] not in node_info:
            node_info[edge[1]] = 1
            adj_dict[edge[1]] = []
        else:
            node_info[edge[1]] += 1
                                            
    for i in node_info.items():
        if i[1] <= -2:
            center_node = i[0]
            num_of_graphs = len(adj_dict[i[0]])
            break
    # center_node의 adjacent node에서 각각 dfs를 실행하면 된다. 
    # center_node의 나가는 간선 개수만큼 그래프가 있다. 각각의 종류만 판단하면 된다.
    def determine_type(node_history:list):
        current_node = node_history[-1]
        next_nodes = adj_dict[current_node]
        if not next_nodes: # 인접노드가 없는 노드가 그래프 내에 존재한다면
            return 2 # 막대 그래프 +1
        elif len(next_nodes) == 2: # 나가는 간선이 2개 있는 정점이 있다면 8자그래프
            return 3 # 8자 그래프 +1
        elif len(next_nodes) == 1 and next_nodes[0] == node_history[0]:
            return 1 # 도넛 그래프
        else:
            node_history.append(next_nodes[0])
            type_code = determine_type(node_history)
            node_history.pop(-1)
            return type_code
        
    for adj_node in adj_dict[center_node]:
        type_code = determine_type([adj_node])
        answer[type_code] += 1
    answer[0] = center_node
    return answer
```

#### 더 효율적으로 풀 수 있는가?

애초에 DFS를 쓸 필요가 없기도 하다. 왜냐하면 adj_list를 확인하는 것 만으로도 "막대 그래프"와 8자그래프의 숫자를 알 수 있고,
그래프의 유형은 3개뿐인데 "추가된 정점"의 간선 수 == 전체 그래프 수 이므로 도넛 그래프의 수는 그냥 간단한 뺄셈만 해주면 파악할 수 있기 때문이다. 

1. out=0인 노드는 막대그래프마다 1개씩 있다.

2. out=2인 노드는 8자그래프마다 1개씩 있다.

즉 다음과 같은 코드로도 풀 수 있다. 

```python
def solution(edges):
    node_info = {}
    adj_dict = {}
    answer = [0, 0, 0, 0]
    # 진입차수가 -2 이하인 노드 = 생성한 정점 
    for edge in edges:
        if edge[0] not in node_info:
            node_info[edge[0]] = -1
            adj_dict[edge[0]] = [edge[1]]
        else: 
            node_info[edge[0]] -= 1
            adj_dict[edge[0]].append(edge[1])
            
        if edge[1] not in node_info:
            node_info[edge[1]] = 1
            adj_dict[edge[1]] = []
        else:
            node_info[edge[1]] += 1
                                            
    for i in node_info.items():
        if i[1] <= -2:
            center_node = i[0]
            num_of_graphs = len(adj_dict[i[0]])
            break
    
    answer = [center_node, 0, 0, 0]
    # 등재된 인접 노드 정보를 보면서 막대그래프와 8자 그래프 개수 파악
    found_graphs = 0
    for i in adj_dict.items():
        if i[1] == []: # 인접노드가 없는 노드 개수 == 막대그래프 개수
            answer[2] += 1
            found_graphs += 1
        elif len(i[1]) == 2: # 2개와 인접하는 노드 개수 == 8자 그래프 개수
            if i[0] == center_node:
                continue
            answer[3] += 1
            found_graphs += 1
           
    answer[1] = num_of_graphs - found_graphs # 도넛그래프 개수는 (전체그래프수 - 막대그래프수 - 8자그래프수) 이다. 
    
    return answer            
```

    