---
layout: post
title:  "[Programmers Lv.2] 더 맵게"
date:   2024-03-09 19:15:16 +0900
categories: study
tags: codingtest heapq
---

level 2 
https://school.programmers.co.kr/learn/courses/30/parts/12117  
Time Spent: 26분  

scoville이 매번 정렬을 유지한다는 가정하에 0번째 원소가 K이상이 되면 바로 종료  
bisect를 할 수도 있을 것 같고 heapq를 쓸 수도 있을 것  
ㄴ 생각해보니 list 길이가 고정되어 있는게 아니라서 자원이 더 많이 소모될 것. (매번 리스트 잘라가면서 새로 붙여서 만들어야 함)  

heapq 메서드 이름 및 사용법에서 조금 헤맴. Minheap을 max-Heap처럼 쓰는 법 복습 필요   
maxheap은 음수로 넣어주면 된다   
heap[1] 은 2번째로 작은 값이 아니다. heapq.nlargest(n, iterable)로 구할 수 있다.  

```python
def solution1(scoville, K):
    import heapq
    # heapq 
    turns = 0
    heapq.heapify(scoville)
    
    while len(scoville) >= 2:
        food1 = heapq.heappop(scoville)
        if food1 >= K:
            return turns
        
        food2 = heapq.heappop(scoville)
        newfood = food1 + (2 * food2)
        heapq.heappush(scoville, newfood)
        turns += 1
        
    if sum(scoville) >= K:
        return turns
    else:
        return -1

if __name__ == '__main__':
    case1, K1 = [1, 2, 3, 9, 10, 12], 7
    print(solution1(case1, K1)) 
```